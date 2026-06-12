"""
model_1d.py

Reusable 1D GAM-FNO model module.

Architecture:
    active nodes -> radial encoder -> regular latent grid -> latent backbone
    -> radial decoder -> hard Dirichlet BC

The model expects normalized coordinates X in [0, 1]. It is intentionally
physics-agnostic: each notebook/problem should define its own energy/residual
loss, material law, loads, quadrature/integration, and validation metrics.

Important contract:
    The batch axis represents multiple load/parameter cases on the same node set
    X. It does not represent multiple meshes/geometries. For multiple meshes,
    loop over geometries or pad them to a common node set.

Supported latent backbones:
    "fno" : Fourier neural operator latent mixer
    "cnn" : local convolutional latent mixer
    "mlp" : parameter-efficient MLP-Mixer latent mixer

Boundary modes:
    "scatter"        : exact nodal Dirichlet scatter only
    "envelope"       : coordinate-aware scalar envelope; supports one-sided or
                       two-ended Dirichlet constraints on the domain boundary
    "envelope_left"  : one-sided scalar envelope fixed at the left boundary
    "envelope_right" : one-sided scalar envelope fixed at the right boundary
    "none"           : no hard BC enforcement

Envelope/output scaling note:
    output_scale is applied to the learned correction. For envelope BC modes,
    output_shift is folded into the coordinate lift so that additive baselines
    are not attenuated by the envelope multiplier. Boundary values should be
    supplied in physical/output units. Envelope modes use data-dependent
    coordinate checks and are not compatible with torch.compile(fullgraph=True).

Radial kernel note:
    The Gaussian kernel is exp(-r^2 / sigma^2). Thus sigma is a bandwidth
    parameter, not the statistical standard deviation; std = sigma / sqrt(2).

Plasticity / history-dependent materials:
    This core model is feed-forward. For plasticity, carry history variables in
    the node features c and use StatefulIncrementalWrapper or an external
    load-stepping wrapper that performs the material return mapping/state update.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor

__version__ = "2.0.1"


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _get_act(name: str):
    name = name.lower()
    if name == "gelu":
        return F.gelu
    if name == "relu":
        return F.relu
    if name == "tanh":
        return torch.tanh
    if name in {"silu", "swish"}:
        return F.silu
    raise ValueError(f"Unknown activation: {name}")


def _as_modes_list(modes: Union[int, Sequence[int]], n_layers_minus_1: int) -> list[int]:
    if isinstance(modes, int):
        return [int(modes)] * n_layers_minus_1
    modes = [int(m) for m in modes]
    if len(modes) != n_layers_minus_1:
        raise ValueError(
            f"Expected {n_layers_minus_1} Fourier mode entries, got {len(modes)}."
        )
    return modes


def _as_channel_buffer(value: Union[float, Sequence[float], Tensor], out_dim: int) -> Tensor:
    t = torch.as_tensor(value, dtype=torch.float32)
    if t.ndim == 0:
        t = t.repeat(out_dim)
    t = t.reshape(-1)
    if t.numel() != out_dim:
        raise ValueError(f"Expected scalar or {out_dim} output-scaling values, got {t.numel()}.")
    return t


def count_trainable_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def compl_mul1d(a: Tensor, b: Tensor) -> Tensor:
    """Complex multiplication for 1D Fourier coefficients.

    Args:
        a: [B, C_in, K]
        b: [C_in, C_out, K]

    Returns:
        [B, C_out, K]
    """
    return torch.einsum("bik,iok->bok", a, b)


# -----------------------------------------------------------------------------
# FNO1D backbone
# -----------------------------------------------------------------------------

class SpectralConv1d(nn.Module):
    """1D spectral convolution layer.

    The number of active modes is clamped to the available one-sided rFFT length
    N//2 + 1 at runtime. Complex weights are cast to the live FFT dtype, so the
    layer works after model.float() and model.double().
    """

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes = int(modes)
        if self.modes < 0:
            raise ValueError("modes must be non-negative.")

        scale = 1.0 / max(1, in_channels * out_channels)
        self.weights = nn.Parameter(
            scale
            * torch.randn(
                self.in_channels,
                self.out_channels,
                max(1, self.modes),
                dtype=torch.cfloat,
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, C_in, N]
        batch_size = x.shape[0]
        n = x.shape[-1]
        x_ft = torch.fft.rfft(x, dim=-1)

        # rFFT has N//2 + 1 usable coefficients. This prevents mode overflow.
        k = min(self.modes, x_ft.shape[-1])
        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            x_ft.shape[-1],
            device=x.device,
            dtype=x_ft.dtype,
        )
        if k > 0:
            out_ft[:, :, :k] = compl_mul1d(
                x_ft[:, :, :k],
                self.weights[:, :, :k].to(dtype=x_ft.dtype, device=x.device),
            )
        return torch.fft.irfft(out_ft, n=n, dim=-1)


class FNO1d(nn.Module):
    """Vanilla 1D FNO block used as the latent-grid mixer.

    Args:
        padding: optional right-side latent padding before spectral layers and
            crop after them. This mitigates periodic wrap-around for bounded,
            non-periodic fields. padding=0 preserves the original behavior.
    """

    def __init__(
        self,
        modes: Union[int, Sequence[int]],
        width: int = 64,
        layers: Optional[Sequence[int]] = None,
        fc_dim: int = 128,
        in_dim: int = 1,
        out_dim: int = 1,
        act: str = "gelu",
        padding: int = 0,
    ):
        super().__init__()
        if layers is None:
            layers = [width] * 4
        layers = list(layers)
        if len(layers) < 2:
            raise ValueError("layers must contain at least two channel widths.")
        if padding < 0:
            raise ValueError("padding must be non-negative.")

        modes_list = _as_modes_list(modes, len(layers) - 1)

        self.fc0 = nn.Linear(in_dim, layers[0])
        self.sp_convs = nn.ModuleList(
            SpectralConv1d(cin, cout, m)
            for cin, cout, m in zip(layers[:-1], layers[1:], modes_list)
        )
        self.ws = nn.ModuleList(
            nn.Conv1d(cin, cout, kernel_size=1)
            for cin, cout in zip(layers[:-1], layers[1:])
        )
        self.fc1 = nn.Linear(layers[-1], fc_dim)
        self.fc2 = nn.Linear(fc_dim, out_dim)
        self.act = _get_act(act)
        self.padding = int(padding)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, N_L, C]
        x = self.fc0(x)
        x = x.permute(0, 2, 1)  # [B, C, N_L]

        original_n = x.shape[-1]
        if self.padding > 0:
            x = F.pad(x, (0, self.padding))

        n_blocks = len(self.ws)
        for i, (spectral, pointwise) in enumerate(zip(self.sp_convs, self.ws)):
            x = spectral(x) + pointwise(x)
            if i != n_blocks - 1:
                x = self.act(x)

        if self.padding > 0:
            x = x[..., :original_n]

        x = x.permute(0, 2, 1)  # [B, N_L, C]
        x = self.act(self.fc1(x))
        return self.fc2(x)


# -----------------------------------------------------------------------------
# Alternative latent backbones
# -----------------------------------------------------------------------------

class CNN1dBulk(nn.Module):
    """Residual local convolutional mixer on the 1D latent grid."""

    def __init__(self, dim: int, width: int = 64, depth: int = 4, kernel_size: int = 5):
        super().__init__()
        if depth < 2:
            raise ValueError("CNN depth must be at least 2.")
        pad = kernel_size // 2

        layers: list[nn.Module] = [nn.Conv1d(dim, width, kernel_size, padding=pad), nn.GELU()]
        for _ in range(depth - 2):
            layers += [nn.Conv1d(width, width, kernel_size, padding=pad), nn.GELU()]
        layers += [nn.Conv1d(width, dim, kernel_size, padding=pad)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, N_L, C]
        y = self.net(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x + y


class MLP1dBulk(nn.Module):
    """Parameter-efficient MLP-Mixer style latent backbone."""

    def __init__(
        self,
        n_latent: int,
        dim: int,
        token_hidden: int = 32,
        channel_hidden: int = 64,
        depth: int = 2,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "ln1": nn.LayerNorm(dim),
                        "token": nn.Sequential(
                            nn.Linear(n_latent, token_hidden),
                            nn.GELU(),
                            nn.Linear(token_hidden, n_latent),
                        ),
                        "ln2": nn.LayerNorm(dim),
                        "channel": nn.Sequential(
                            nn.Linear(dim, channel_hidden),
                            nn.GELU(),
                            nn.Linear(channel_hidden, dim),
                        ),
                    }
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, N_L, C]
        for block in self.blocks:
            y = block["ln1"](x).transpose(1, 2)  # [B, C, N_L]
            y = block["token"](y).transpose(1, 2)
            x = x + y
            x = x + block["channel"](block["ln2"](x))
        return x


def make_latent_backbone(
    backbone: str,
    latent_dim: int,
    n_latent: int,
    width: int = 32,
    modes: Union[int, Sequence[int]] = 16,
    layers: int = 5,
    fc_dim: int = 64,
    fno_padding: int = 0,
) -> nn.Module:
    """Factory for the 1D latent backbone."""
    backbone = backbone.lower()
    if layers < 2:
        raise ValueError("layers must be at least 2.")

    if backbone == "fno":
        return FNO1d(
            modes=modes,
            width=width,
            layers=[width] * layers,
            fc_dim=fc_dim,
            in_dim=latent_dim,
            out_dim=latent_dim,
            act="gelu",
            padding=fno_padding,
        )
    if backbone == "cnn":
        return CNN1dBulk(dim=latent_dim, width=width, depth=layers, kernel_size=5)
    if backbone == "mlp":
        return MLP1dBulk(
            n_latent=n_latent,
            dim=latent_dim,
            token_hidden=width,
            channel_hidden=fc_dim,
            depth=max(1, layers - 2),
        )
    raise ValueError("backbone must be one of: 'fno', 'cnn', 'mlp'.")


# -----------------------------------------------------------------------------
# Radial encoder / decoder
# -----------------------------------------------------------------------------

class RadialEncoder1D(nn.Module):
    """Encode active-node features onto a regular 1D latent grid."""

    def __init__(
        self,
        node_in_dim: int,
        latent_dim: int,
        hidden: int = 64,
        sigma: float = 0.08,
        normalize: bool = True,
        chunk_size: Optional[int] = None,
        support_eps: float = 1e-12,
        support_threshold: float = 0.0,
    ):
        super().__init__()
        self.sigma = float(sigma)
        self.normalize = bool(normalize)
        self.chunk_size = chunk_size
        self.support_eps = float(support_eps)
        self.support_threshold = float(support_threshold)
        if self.sigma <= 0:
            raise ValueError("sigma must be positive.")
        self.phi = nn.Sequential(
            nn.Linear(node_in_dim + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, latent_dim),
        )

    def _encode_chunk(self, X: Tensor, c: Tensor, P_chunk: Tensor) -> Tensor:
        B, N, C = c.shape
        M = P_chunk.shape[0]
        dx = X[:, 0][None, None, :] - P_chunk[:, 0][None, :, None]  # [1, M, N]
        w = torch.exp(-((dx / self.sigma) ** 2))
        support = w.sum(dim=-1, keepdim=True)
        if self.normalize:
            w = w / support.clamp_min(self.support_eps)
        if self.support_threshold > 0.0:
            w = torch.where(support > self.support_threshold, w, torch.zeros_like(w))

        c_expand = c[:, None, :, :].expand(B, M, N, C)
        dx_expand = dx[..., None].expand(B, M, N, 1)
        msg = self.phi(torch.cat([dx_expand, c_expand], dim=-1))
        return torch.sum(w[..., None] * msg, dim=2)

    def support(self, X: Tensor, P: Tensor) -> Tensor:
        X = X.to(device=P.device, dtype=P.dtype)
        dx = X[:, 0][None, :] - P[:, 0][:, None]
        return torch.exp(-((dx / self.sigma) ** 2)).sum(dim=-1)

    def support_mask(self, X: Tensor, P: Tensor, threshold: Optional[float] = None) -> Tensor:
        threshold = self.support_threshold if threshold is None else float(threshold)
        return self.support(X, P) > threshold

    def forward(self, X: Tensor, c: Tensor, P: Tensor) -> Tensor:
        """Forward pass.

        Args:
            X: normalized active-node coordinates, [N, 1]
            c: node features, [N, C] or [B, N, C]
            P: latent coordinates, [N_L, 1]

        Returns:
            latent features, [B, N_L, latent_dim]
        """
        if c.ndim == 2:
            c = c.unsqueeze(0)
        if X.ndim != 2 or X.shape[-1] != 1:
            raise ValueError("X must have shape [N, 1].")
        if c.ndim != 3:
            raise ValueError("c must have shape [N, C] or [B, N, C].")

        X = X.to(device=P.device, dtype=P.dtype)
        c = c.to(device=P.device, dtype=P.dtype)
        B, N, _ = c.shape
        if X.shape[0] != N:
            raise ValueError(f"X has {X.shape[0]} nodes but c has {N} nodes.")

        if self.chunk_size is None:
            return self._encode_chunk(X, c, P)

        chunks = []
        for start in range(0, P.shape[0], self.chunk_size):
            chunks.append(self._encode_chunk(X, c, P[start : start + self.chunk_size]))
        return torch.cat(chunks, dim=1)


class RadialDecoder1D(nn.Module):
    """Decode regular latent-grid features back to active nodes."""

    def __init__(
        self,
        latent_dim: int,
        out_dim: int = 1,
        hidden: int = 64,
        sigma: float = 0.08,
        normalize: bool = True,
        chunk_size: Optional[int] = None,
        support_eps: float = 1e-12,
        support_threshold: float = 0.0,
    ):
        super().__init__()
        self.sigma = float(sigma)
        self.normalize = bool(normalize)
        self.chunk_size = chunk_size
        self.support_eps = float(support_eps)
        self.support_threshold = float(support_threshold)
        self.out_dim = int(out_dim)
        if self.sigma <= 0:
            raise ValueError("sigma must be positive.")
        self.phi = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def _decode_chunk(self, X_chunk: Tensor, h: Tensor, P: Tensor) -> Tensor:
        B, N_L, H = h.shape
        N = X_chunk.shape[0]
        dx = X_chunk[:, 0][None, :, None] - P[:, 0][None, None, :]  # [1, N, N_L]
        w = torch.exp(-((dx / self.sigma) ** 2))
        support = w.sum(dim=-1, keepdim=True)
        if self.normalize:
            w = w / support.clamp_min(self.support_eps)
        if self.support_threshold > 0.0:
            w = torch.where(support > self.support_threshold, w, torch.zeros_like(w))

        h_expand = h[:, None, :, :].expand(B, N, N_L, H)
        dx_expand = dx[..., None].expand(B, N, N_L, 1)
        msg = self.phi(torch.cat([h_expand, dx_expand], dim=-1))
        return torch.sum(w[..., None] * msg, dim=2)  # [B, N, out_dim]

    def forward(self, X: Tensor, h: Tensor, P: Tensor) -> Tensor:
        """Forward pass.

        Args:
            X: normalized active-node coordinates, [N, 1]
            h: latent features, [B, N_L, latent_dim]
            P: latent coordinates, [N_L, 1]

        Returns:
            nodal output, [B, N, out_dim] if out_dim > 1, otherwise [B, N]
        """
        if X.ndim != 2 or X.shape[-1] != 1:
            raise ValueError("X must have shape [N, 1].")
        if h.ndim != 3:
            raise ValueError("h must have shape [B, N_L, C].")

        X = X.to(device=P.device, dtype=P.dtype)
        h = h.to(device=P.device, dtype=P.dtype)
        if P.shape[0] != h.shape[1]:
            raise ValueError(f"P has {P.shape[0]} latent nodes but h has {h.shape[1]}.")

        if self.chunk_size is None:
            u = self._decode_chunk(X, h, P)
        else:
            chunks = []
            for start in range(0, X.shape[0], self.chunk_size):
                chunks.append(self._decode_chunk(X[start : start + self.chunk_size], h, P))
            u = torch.cat(chunks, dim=1)
        return u[..., 0] if self.out_dim == 1 else u


# -----------------------------------------------------------------------------
# Final 1D model
# -----------------------------------------------------------------------------

class GAMFNO1D(nn.Module):
    """Reusable 1D Geometry-Adaptive/Geometry-Aware FNO model.

    Args:
        node_in_dim: number of node feature channels.
        out_dim: number of output channels. Use 1 for scalar displacement.
        backbone: "fno", "cnn", or "mlp".
        latent_dim: latent channel dimension.
        n_latent: number of regular latent nodes.
        radial_hidden: hidden width of radial encoder/decoder MLPs.
        sigma_enc/sigma_dec: radial kernel bandwidths in normalized coordinates.
        width: latent backbone width.
        modes: Fourier modes for FNO backbone.
        layers: number of latent backbone layers/channels.
        fc_dim: projection/channel hidden width.
        fno_padding: optional right padding inside the FNO latent backbone.
        bc_mode: "envelope", "envelope_left", "envelope_right", "scatter", or "none".
        output_scale/output_shift: optional per-output affine transform for
            nondimensionalized multiphysics fields. For scatter/none BC modes,
            the full affine transform is applied before BC enforcement. For
            envelope BC modes, output_scale multiplies the learned correction
            and output_shift is folded into the lift, avoiding attenuation of
            additive baselines by the envelope multiplier.
        encoder_chunk_size/decoder_chunk_size: optional chunk sizes for radial
            aggregation.
    """

    def __init__(
        self,
        node_in_dim: int,
        out_dim: int = 1,
        backbone: str = "fno",
        latent_dim: int = 32,
        n_latent: int = 64,
        radial_hidden: int = 64,
        sigma_enc: float = 0.08,
        sigma_dec: float = 0.08,
        width: int = 32,
        modes: Union[int, Sequence[int]] = 16,
        layers: int = 5,
        fc_dim: int = 64,
        fno_padding: int = 0,
        bc_mode: str = "envelope",
        output_scale: Union[float, Sequence[float], Tensor] = 1.0,
        output_shift: Union[float, Sequence[float], Tensor] = 0.0,
        encoder_chunk_size: Optional[int] = None,
        decoder_chunk_size: Optional[int] = None,
        radial_support_eps: float = 1e-12,
    ):
        super().__init__()
        self.backbone_name = backbone.lower()
        self.bc_mode = bc_mode.lower()
        self.out_dim = int(out_dim)
        self.n_latent = int(n_latent)

        self.register_buffer("P", torch.linspace(0.0, 1.0, n_latent).view(-1, 1))
        self.register_buffer("output_scale", _as_channel_buffer(output_scale, self.out_dim))
        self.register_buffer("output_shift", _as_channel_buffer(output_shift, self.out_dim))

        self.encoder = RadialEncoder1D(
            node_in_dim=node_in_dim,
            latent_dim=latent_dim,
            hidden=radial_hidden,
            sigma=sigma_enc,
            normalize=True,
            chunk_size=encoder_chunk_size,
            support_eps=radial_support_eps,
        )
        self.bulk = make_latent_backbone(
            backbone=self.backbone_name,
            latent_dim=latent_dim,
            n_latent=n_latent,
            width=width,
            modes=modes,
            layers=layers,
            fc_dim=fc_dim,
            fno_padding=fno_padding,
        )
        self.decoder = RadialDecoder1D(
            latent_dim=latent_dim,
            out_dim=out_dim,
            hidden=radial_hidden,
            sigma=sigma_dec,
            normalize=True,
            chunk_size=decoder_chunk_size,
            support_eps=radial_support_eps,
        )

    def _apply_output_scale(self, raw: Tensor) -> Tensor:
        scale = self.output_scale.to(device=raw.device, dtype=raw.dtype)
        if self.out_dim == 1:
            return raw * scale[0]
        return raw * scale.view(1, 1, -1)

    def _apply_output_transform(self, raw: Tensor) -> Tensor:
        scaled = self._apply_output_scale(raw)
        shift = self.output_shift.to(device=raw.device, dtype=raw.dtype)
        if self.out_dim == 1:
            return scaled + shift[0]
        return scaled + shift.view(1, 1, -1)

    def _active_bc_mode(self, bc_mode: Optional[str]) -> str:
        return self.bc_mode if bc_mode is None else bc_mode.lower()

    def _prepare_dirichlet(
        self,
        raw: Tensor,
        dirichlet_idx: Optional[Union[Tensor, Sequence[int]]],
        dirichlet_val: Optional[Union[Tensor, Sequence[float]]],
    ) -> tuple[Optional[Tensor], Optional[Tensor]]:
        if dirichlet_idx is None and dirichlet_val is None:
            return None, None
        if dirichlet_idx is None or dirichlet_val is None:
            raise ValueError("dirichlet_idx and dirichlet_val must be provided together.")

        idx = torch.as_tensor(dirichlet_idx, device=raw.device, dtype=torch.long).reshape(-1)
        vals = torch.as_tensor(dirichlet_val, device=raw.device, dtype=raw.dtype)
        if self.out_dim == 1:
            vals = vals.reshape(-1)
            if vals.numel() != idx.numel():
                raise ValueError(
                    f"Scalar dirichlet_val must contain {idx.numel()} values, got {vals.numel()}."
                )
        else:
            vals = vals.reshape(idx.numel(), self.out_dim)
        return idx, vals

    def hard_bc(
        self,
        X: Tensor,
        raw: Tensor,
        dirichlet_idx: Optional[Union[Tensor, Sequence[int]]] = None,
        dirichlet_val: Optional[Union[Tensor, Sequence[float]]] = None,
        bc_mode: Optional[str] = None,
    ) -> Tensor:
        """Apply hard Dirichlet boundary conditions.

        Coordinate-aware envelope behavior for scalar outputs:
            * 2 boundary nodes: values are assigned to left/right by coordinate,
              not by user-provided order.
            * 1 boundary node: one-sided envelope, inferred from whether the node
              is at the left or right domain boundary.

        A final exact scatter is applied for all modes except "none".
        """
        mode = self.bc_mode if bc_mode is None else bc_mode.lower()
        allowed = {"envelope", "envelope_left", "envelope_right", "scatter", "none"}
        if mode not in allowed:
            raise ValueError(f"bc_mode must be one of {sorted(allowed)}, got {mode!r}.")

        X = X.to(device=raw.device, dtype=raw.dtype)
        u = raw
        idx, vals = self._prepare_dirichlet(raw, dirichlet_idx, dirichlet_val)

        if mode.startswith("envelope"):
            if self.out_dim != 1:
                raise ValueError("Envelope BC modes currently support out_dim=1 only.")
            if idx is None:
                # Do not silently impose u(0)=u(1)=0 when no BC is given.
                u = raw
            else:
                x = X[:, 0]
                x_min = x.min()
                x_max = x.max()
                tol = max(1e-6, 10.0 * torch.finfo(raw.dtype).eps)

                if idx.numel() == 2:
                    if mode in {"envelope_left", "envelope_right"}:
                        raise ValueError(f"{mode} expects exactly one Dirichlet node.")
                    x_bc = x[idx]
                    order = torch.argsort(x_bc)
                    idx_sorted = idx[order]
                    vals_sorted = vals[order]
                    x_left = x[idx_sorted[0]]
                    x_right = x[idx_sorted[1]]
                    if torch.abs(x_left - x_min) > tol or torch.abs(x_right - x_max) > tol:
                        raise ValueError(
                            "Two-point envelope requires the constrained nodes to be the left and right domain boundaries. "
                            "Use bc_mode='scatter' for interior/multi-point Dirichlet constraints."
                        )
                    denom = (x_right - x_left).clamp_min(tol)
                    v_left = vals_sorted[0]
                    v_right = vals_sorted[1]
                    lift = ((x_right - x) / denom) * v_left + ((x - x_left) / denom) * v_right
                    phi = ((x - x_left) * (x_right - x)) / (denom * denom)
                    u = lift[None, :] + phi[None, :] * raw

                elif idx.numel() == 1:
                    x0 = x[idx[0]]
                    v0 = vals[0]
                    is_left = torch.abs(x0 - x_min) <= tol
                    is_right = torch.abs(x0 - x_max) <= tol
                    if mode == "envelope_left" and not bool(is_left):
                        raise ValueError("bc_mode='envelope_left' requires the constrained node at the left boundary.")
                    if mode == "envelope_right" and not bool(is_right):
                        raise ValueError("bc_mode='envelope_right' requires the constrained node at the right boundary.")
                    if mode == "envelope" and not (bool(is_left) or bool(is_right)):
                        raise ValueError(
                            "One-sided envelope requires the single constrained node to be at the left or right boundary. "
                            "Use bc_mode='scatter' for interior Dirichlet constraints."
                        )
                    if bool(is_left):
                        denom = (x_max - x0).clamp_min(tol)
                        phi = (x - x0) / denom
                    else:
                        denom = (x0 - x_min).clamp_min(tol)
                        phi = (x0 - x) / denom
                    lift = torch.full_like(x, v0)
                    u = lift[None, :] + phi[None, :] * raw
                else:
                    raise ValueError(
                        "Envelope BC supports one boundary node or two end-boundary nodes. "
                        "Use bc_mode='scatter' for general/multi-point Dirichlet constraints."
                    )

        elif mode in {"scatter", "none"}:
            u = raw

        if mode != "none" and idx is not None:
            u = u.clone()
            if self.out_dim == 1:
                u[:, idx] = vals.reshape(1, -1)
            else:
                u[:, idx, :] = vals.reshape(1, len(idx), self.out_dim)
        return u

    def forward(
        self,
        X: Tensor,
        c: Tensor,
        dirichlet_idx: Optional[Union[Tensor, Sequence[int]]] = None,
        dirichlet_val: Optional[Union[Tensor, Sequence[float]]] = None,
        bc_mode: Optional[str] = None,
        return_dict: bool = False,
    ):
        """Run the model.

        Args:
            X: normalized coordinates, [N, 1].
            c: node features, [N, C] or [B, N, C]. Batch means same X.
            dirichlet_idx: optional Dirichlet node indices, [N_D].
            dirichlet_val: optional Dirichlet values, [N_D] for scalar output or
                [N_D, out_dim] for vector output.
            bc_mode: optional override of self.bc_mode.
            return_dict: if True, return latent tensors and raw/scattered outputs.

        Returns:
            u if return_dict=False. Shape is [N] for unbatched scalar input,
            [B, N] for batched scalar input, or with trailing out_dim for vector output.
        """
        unbatched = c.ndim == 2
        X = X.to(device=self.P.device, dtype=self.P.dtype)
        c = c.to(device=self.P.device, dtype=self.P.dtype)

        h0 = self.encoder(X, c, self.P)
        h = self.bulk(h0)
        raw_net = self.decoder(X, h, self.P)

        mode = self._active_bc_mode(bc_mode)
        has_dirichlet = dirichlet_idx is not None or dirichlet_val is not None

        # For scatter/none, the unconstrained field is the full affine output.
        # For envelope BCs, the learned term is a correction multiplied by the
        # envelope. Applying output_shift before the envelope would attenuate
        # the additive baseline by phi(x), so only output_scale is applied to
        # the correction. The physical baseline is supplied by the coordinate
        # lift constructed from dirichlet_val.
        raw_physical = self._apply_output_transform(raw_net)
        if mode.startswith("envelope") and has_dirichlet:
            raw_for_bc = self._apply_output_scale(raw_net)
        else:
            raw_for_bc = raw_physical

        u = self.hard_bc(X, raw_for_bc, dirichlet_idx, dirichlet_val, bc_mode)

        if unbatched:
            u_out = u[0]
            raw_out = raw_physical[0]
        else:
            u_out = u
            raw_out = raw_physical

        if return_dict:
            return {
                "latent_in": h0,
                "latent": h,
                "u_raw": raw_out,
                "u": u_out,
            }
        return u_out

    def count_params(self) -> int:
        return count_trainable_params(self)


class StatefulIncrementalWrapper(nn.Module):
    """Thin wrapper for history-dependent or incremental physics.

    The wrapped model must be constructed with node_in_dim equal to
    base_feature_dim + state_dim. At each step, this wrapper concatenates the
    current state variables to c and calls the feed-forward model.

    A problem-specific state_update_fn may be provided, for example a return-map
    update for plastic strain/hardening variables. Signature:
        state_next = state_update_fn(X, c, state, u, **kwargs)

    This wrapper intentionally does not implement a constitutive law; it only
    standardizes the data contract for plasticity/multistep notebooks.
    """

    def __init__(
        self,
        model: GAMFNO1D,
        state_update_fn: Optional[Callable[..., Tensor]] = None,
    ):
        super().__init__()
        self.model = model
        self.state_update_fn = state_update_fn

    def forward(
        self,
        X: Tensor,
        c: Tensor,
        state: Tensor,
        *model_args,
        return_state: bool = True,
        **model_kwargs,
    ):
        c_aug = torch.cat([c, state], dim=-1)
        u = self.model(X, c_aug, *model_args, **model_kwargs)
        if not return_state:
            return u
        state_next = None if self.state_update_fn is None else self.state_update_fn(X, c, state, u)
        return u, state_next


__all__ = [
    "GAMFNO1D",
    "StatefulIncrementalWrapper",
    "RadialEncoder1D",
    "RadialDecoder1D",
    "FNO1d",
    "SpectralConv1d",
    "CNN1dBulk",
    "MLP1dBulk",
    "make_latent_backbone",
    "count_trainable_params",
]
