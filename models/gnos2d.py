"""
model_2d.py

Reusable 2D GAM-FNO model module.

Architecture:
    active nodes -> radial encoder -> regular latent grid -> latent backbone
    -> radial decoder -> hard Dirichlet scatter / optional geometry mask

The model expects normalized coordinates X in [0, 1]^2. It is intentionally
physics-agnostic: each notebook/problem should define its own weak-form or
strong-form physics loss, material law, loads, quadrature/integration, and
validation metrics.

Important contract:
    The batch axis represents multiple load/parameter cases on the same node set
    X. It does not represent multiple meshes/geometries. For multiple meshes,
    loop over geometries or pad them to a common node set.

Supported latent backbones:
    "fno" : Fourier neural operator latent mixer
    "cnn" : local convolutional latent mixer
    "mlp" : parameter-efficient 2D MLP-Mixer latent mixer

Radial kernel note:
    The Gaussian kernel is exp(-r^2 / sigma^2). Thus sigma is a bandwidth
    parameter, not the statistical standard deviation; std = sigma / sqrt(2).

Latent support note:
    latent_support_threshold controls which latent-grid points are considered
    unsupported by the active geometry. The default 1e-10 is conservative, but
    this threshold is geometry- and sigma-sensitive and should be tuned for
    holes, notches, thin members, or strongly non-rectangular point clouds.

Plasticity / history-dependent materials:
    This core model is feed-forward. For plasticity, carry history variables in
    the node features c and use StatefulIncrementalWrapper or an external
    load-stepping wrapper that performs the material return mapping/state update.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple, Union

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


def _pair(x: Union[int, Sequence[int]]) -> tuple[int, int]:
    if isinstance(x, int):
        return int(x), int(x)
    x = tuple(x)
    if len(x) != 2:
        raise ValueError("Expected an int or a length-2 sequence.")
    return int(x[0]), int(x[1])


def _as_mode_list(
    modes: Union[int, Sequence[int]],
    n_layers_minus_1: int,
    name: str,
) -> list[int]:
    if isinstance(modes, int):
        return [int(modes)] * n_layers_minus_1
    modes = [int(m) for m in modes]
    if len(modes) != n_layers_minus_1:
        raise ValueError(
            f"Expected {n_layers_minus_1} {name} entries, got {len(modes)}."
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


def make_latent_grid(n_latent: tuple[int, int], device=None, dtype=None) -> Tensor:
    """Create normalized latent grid coordinates.

    Returns:
        P: [NLy, NLx, 2] with columns [x, y].
    """
    ny, nx = int(n_latent[0]), int(n_latent[1])
    y = torch.linspace(0.0, 1.0, ny, device=device, dtype=dtype)
    x = torch.linspace(0.0, 1.0, nx, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([xx, yy], dim=-1)


def compl_mul2d(a: Tensor, b: Tensor) -> Tensor:
    """Complex multiplication for 2D Fourier coefficients.

    Args:
        a: [B, C_in, K1, K2]
        b: [C_in, C_out, K1, K2]

    Returns:
        [B, C_out, K1, K2]
    """
    return torch.einsum("bixy,ioxy->boxy", a, b)


def hard_scatter_bc(u_raw: Tensor, bc_mask: Optional[Tensor], bc_val: Optional[Tensor]) -> Tensor:
    """Apply exact nodal Dirichlet values using torch.where.

    Args:
        u_raw: [B, N, D] or [N, D]
        bc_mask: bool tensor broadcastable to u_raw.
        bc_val: tensor broadcastable to u_raw.
    """
    if bc_mask is None or bc_val is None:
        return u_raw
    return torch.where(
        bc_mask.to(device=u_raw.device, dtype=torch.bool),
        bc_val.to(device=u_raw.device, dtype=u_raw.dtype),
        u_raw,
    )


# -----------------------------------------------------------------------------
# FNO2D backbone
# -----------------------------------------------------------------------------

class SpectralConv2d(nn.Module):
    """2D spectral convolution layer with two-sided modes in the first axis.

    The first-axis modes are clamped to H//2 at runtime to avoid overlap between
    the positive slice [:m1] and negative slice [-m1:]. The second-axis modes are
    clamped to the one-sided rFFT length W//2 + 1. Complex weights are cast to the
    live FFT dtype, so the layer works after model.float() and model.double().
    """

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)
        if self.modes1 < 0 or self.modes2 < 0:
            raise ValueError("modes1 and modes2 must be non-negative.")

        scale = 1.0 / max(1, in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale
            * torch.randn(
                self.in_channels,
                self.out_channels,
                max(1, self.modes1),
                max(1, self.modes2),
                dtype=torch.cfloat,
            )
        )
        self.weights2 = nn.Parameter(
            scale
            * torch.randn(
                self.in_channels,
                self.out_channels,
                max(1, self.modes1),
                max(1, self.modes2),
                dtype=torch.cfloat,
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, C_in, H, W]
        batch_size = x.shape[0]
        H, W = x.shape[-2], x.shape[-1]
        x_ft = torch.fft.rfftn(x, dim=(-2, -1))

        m1 = min(self.modes1, H // 2)          # prevents two-sided overlap
        m2 = min(self.modes2, W // 2 + 1)      # one-sided rFFT length
        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            H,
            W // 2 + 1,
            device=x.device,
            dtype=x_ft.dtype,
        )

        if m1 > 0 and m2 > 0:
            w1 = self.weights1[:, :, :m1, :m2].to(device=x.device, dtype=x_ft.dtype)
            w2 = self.weights2[:, :, :m1, :m2].to(device=x.device, dtype=x_ft.dtype)

            out_ft[:, :, :m1, :m2] = compl_mul2d(x_ft[:, :, :m1, :m2], w1)
            out_ft[:, :, -m1:, :m2] = compl_mul2d(x_ft[:, :, -m1:, :m2], w2)
        return torch.fft.irfftn(out_ft, s=(H, W), dim=(-2, -1))


class FNO2d(nn.Module):
    """Vanilla 2D FNO block used as the latent-grid mixer.

    Args:
        padding: optional right/bottom latent padding before spectral layers and
            crop after them. This mitigates periodic wrap-around for bounded,
            non-periodic fields. padding=0 preserves the original behavior.
    """

    def __init__(
        self,
        modes1: Union[int, Sequence[int]],
        modes2: Union[int, Sequence[int]],
        width: int = 64,
        layers: Optional[Sequence[int]] = None,
        fc_dim: int = 128,
        in_dim: int = 32,
        out_dim: int = 32,
        act: str = "gelu",
        padding: Union[int, Sequence[int]] = 0,
    ):
        super().__init__()
        if layers is None:
            layers = [width] * 4
        layers = list(layers)
        if len(layers) < 2:
            raise ValueError("layers must contain at least two channel widths.")

        self.padding = _pair(padding)
        if self.padding[0] < 0 or self.padding[1] < 0:
            raise ValueError("padding must be non-negative.")

        modes1_list = _as_mode_list(modes1, len(layers) - 1, "modes1")
        modes2_list = _as_mode_list(modes2, len(layers) - 1, "modes2")

        self.fc0 = nn.Linear(in_dim, layers[0])
        self.sp_convs = nn.ModuleList(
            SpectralConv2d(cin, cout, m1, m2)
            for cin, cout, m1, m2 in zip(
                layers[:-1], layers[1:], modes1_list, modes2_list
            )
        )
        self.ws = nn.ModuleList(
            nn.Conv2d(cin, cout, kernel_size=1)
            for cin, cout in zip(layers[:-1], layers[1:])
        )
        self.fc1 = nn.Linear(layers[-1], fc_dim)
        self.fc2 = nn.Linear(fc_dim, out_dim)
        self.act = _get_act(act)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, H, W, C]
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)  # [B, C, H, W]

        original_h, original_w = x.shape[-2], x.shape[-1]
        pad_y, pad_x = self.padding
        if pad_y > 0 or pad_x > 0:
            x = F.pad(x, (0, pad_x, 0, pad_y))

        n_blocks = len(self.ws)
        for i, (spectral, pointwise) in enumerate(zip(self.sp_convs, self.ws)):
            x = spectral(x) + pointwise(x)
            if i != n_blocks - 1:
                x = self.act(x)

        if pad_y > 0 or pad_x > 0:
            x = x[..., :original_h, :original_w]

        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        x = self.act(self.fc1(x))
        return self.fc2(x)


# -----------------------------------------------------------------------------
# Alternative latent backbones
# -----------------------------------------------------------------------------

class CNN2dBulk(nn.Module):
    """Residual local convolutional mixer on the 2D latent grid."""

    def __init__(self, dim: int, width: int = 64, depth: int = 5, kernel_size: int = 5):
        super().__init__()
        if depth < 2:
            raise ValueError("CNN depth must be at least 2.")
        pad = kernel_size // 2
        layers: list[nn.Module] = [nn.Conv2d(dim, width, kernel_size, padding=pad), nn.GELU()]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(width, width, kernel_size, padding=pad), nn.GELU()]
        layers += [nn.Conv2d(width, dim, kernel_size, padding=pad)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, H, W, C]
        y = self.net(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return x + y


class MLP2dBulk(nn.Module):
    """Parameter-efficient MLP-Mixer style latent backbone for 2D grids."""

    def __init__(
        self,
        n_latent: tuple[int, int],
        dim: int,
        token_hidden: int = 64,
        channel_hidden: int = 128,
        depth: int = 2,
    ):
        super().__init__()
        self.n_latent = (int(n_latent[0]), int(n_latent[1]))
        self.n_tokens = self.n_latent[0] * self.n_latent[1]
        self.dim = int(dim)
        self.blocks = nn.ModuleList()

        for _ in range(depth):
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "ln1": nn.LayerNorm(dim),
                        "token": nn.Sequential(
                            nn.Linear(self.n_tokens, token_hidden),
                            nn.GELU(),
                            nn.Linear(token_hidden, self.n_tokens),
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
        # x: [B, H, W, C]
        B, H, W, C = x.shape
        if (H, W) != self.n_latent:
            raise ValueError(
                f"MLP2dBulk was built for latent grid {self.n_latent}, got {(H, W)}."
            )
        x = x.reshape(B, H * W, C)
        for block in self.blocks:
            y = block["ln1"](x).transpose(1, 2)  # [B, C, tokens]
            y = block["token"](y).transpose(1, 2)
            x = x + y
            x = x + block["channel"](block["ln2"](x))
        return x.reshape(B, H, W, C)


def make_latent_backbone_2d(
    backbone: str,
    latent_dim: int,
    n_latent: tuple[int, int],
    width: int = 64,
    modes1: Union[int, Sequence[int]] = 8,
    modes2: Union[int, Sequence[int]] = 8,
    layers: int = 4,
    fc_dim: int = 128,
    fno_padding: Union[int, Sequence[int]] = 0,
) -> nn.Module:
    """Factory for the 2D latent backbone."""
    backbone = backbone.lower()
    if layers < 2:
        raise ValueError("layers must be at least 2.")

    if backbone == "fno":
        return FNO2d(
            modes1=modes1,
            modes2=modes2,
            width=width,
            layers=[width] * layers,
            fc_dim=fc_dim,
            in_dim=latent_dim,
            out_dim=latent_dim,
            act="gelu",
            padding=fno_padding,
        )
    if backbone == "cnn":
        return CNN2dBulk(dim=latent_dim, width=width, depth=layers, kernel_size=5)
    if backbone == "mlp":
        return MLP2dBulk(
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

class RadialEncoder2D(nn.Module):
    """Encode active-node features onto a regular 2D latent grid.

    Low-support latent nodes can be masked to prevent near-empty regions in the
    bounding box from becoming ill-defined features that are later mixed by FNO.
    """

    def __init__(
        self,
        node_in_dim: int,
        latent_dim: int,
        hidden: int = 64,
        sigma: float = 0.08,
        normalize: bool = True,
        chunk_size: Optional[int] = None,
        support_eps: float = 1e-12,
        support_threshold: float = 1e-10,
        zero_low_support: bool = True,
    ):
        super().__init__()
        self.sigma = float(sigma)
        self.normalize = bool(normalize)
        self.chunk_size = chunk_size
        self.support_eps = float(support_eps)
        self.support_threshold = float(support_threshold)
        self.zero_low_support = bool(zero_low_support)
        if self.sigma <= 0:
            raise ValueError("sigma must be positive.")
        self.phi = nn.Sequential(
            nn.Linear(node_in_dim + 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, latent_dim),
        )

    def _support_chunk(self, X: Tensor, P_chunk: Tensor) -> Tensor:
        dr = X[None, :, :] - P_chunk[:, None, :]  # [M, N, 2]
        dist2 = torch.sum(dr * dr, dim=-1)
        return torch.exp(-dist2 / (self.sigma**2)).sum(dim=-1)  # [M]

    def support(self, X: Tensor, P: Tensor) -> Tensor:
        """Return unnormalized Gaussian support for each latent grid node.

        Args:
            X: [N, 2]
            P: [NLy, NLx, 2]

        Returns:
            support: [NLy, NLx]
        """
        X = X.to(device=P.device, dtype=P.dtype)
        P_flat = P.reshape(-1, 2)
        if self.chunk_size is None:
            s = self._support_chunk(X, P_flat)
        else:
            chunks = []
            for start in range(0, P_flat.shape[0], self.chunk_size):
                chunks.append(self._support_chunk(X, P_flat[start : start + self.chunk_size]))
            s = torch.cat(chunks, dim=0)
        return s.reshape(P.shape[0], P.shape[1])

    def support_mask(self, X: Tensor, P: Tensor, threshold: Optional[float] = None) -> Tensor:
        threshold = self.support_threshold if threshold is None else float(threshold)
        return self.support(X, P) > threshold

    def _encode_chunk(self, X: Tensor, c: Tensor, P_chunk: Tensor) -> Tensor:
        B, N, C = c.shape
        M = P_chunk.shape[0]

        dr = X[None, None, :, :] - P_chunk[None, :, None, :]  # [1, M, N, 2]
        dist2 = torch.sum(dr * dr, dim=-1)
        w = torch.exp(-dist2 / (self.sigma**2))
        support = w.sum(dim=-1, keepdim=True)
        if self.normalize:
            w = w / support.clamp_min(self.support_eps)
        if self.zero_low_support and self.support_threshold > 0.0:
            w = torch.where(support > self.support_threshold, w, torch.zeros_like(w))

        c_expand = c[:, None, :, :].expand(B, M, N, C)
        dr_expand = dr.expand(B, M, N, 2)
        msg = self.phi(torch.cat([dr_expand, c_expand], dim=-1))
        return torch.sum(w[..., None] * msg, dim=2)  # [B, M, latent_dim]

    def forward(self, X: Tensor, c: Tensor, P: Tensor) -> Tensor:
        """Forward pass.

        Args:
            X: normalized active-node coordinates, [N, 2].
            c: node features, [N, C] or [B, N, C].
            P: latent grid coordinates, [NLy, NLx, 2].

        Returns:
            latent features, [B, NLy, NLx, latent_dim].
        """
        if c.ndim == 2:
            c = c.unsqueeze(0)
        if X.ndim != 2 or X.shape[-1] != 2:
            raise ValueError("X must have shape [N, 2].")
        if c.ndim != 3:
            raise ValueError("c must have shape [N, C] or [B, N, C].")
        if P.ndim != 3 or P.shape[-1] != 2:
            raise ValueError("P must have shape [NLy, NLx, 2].")

        X = X.to(device=P.device, dtype=P.dtype)
        c = c.to(device=P.device, dtype=P.dtype)
        B, N, _ = c.shape
        if X.shape[0] != N:
            raise ValueError(f"X has {X.shape[0]} nodes but c has {N} nodes.")

        NLy, NLx = P.shape[:2]
        P_flat = P.reshape(-1, 2)
        if self.chunk_size is None:
            h = self._encode_chunk(X, c, P_flat)
        else:
            chunks = []
            for start in range(0, P_flat.shape[0], self.chunk_size):
                chunks.append(self._encode_chunk(X, c, P_flat[start : start + self.chunk_size]))
            h = torch.cat(chunks, dim=1)
        return h.reshape(B, NLy, NLx, -1)


class RadialDecoder2D(nn.Module):
    """Decode regular 2D latent-grid features back to active nodes."""

    def __init__(
        self,
        latent_dim: int,
        out_dim: int = 2,
        hidden: int = 64,
        sigma: float = 0.08,
        normalize: bool = True,
        chunk_size: Optional[int] = None,
        support_eps: float = 1e-12,
        support_threshold: float = 0.0,
        zero_low_support: bool = False,
    ):
        super().__init__()
        self.sigma = float(sigma)
        self.normalize = bool(normalize)
        self.chunk_size = chunk_size
        self.support_eps = float(support_eps)
        self.support_threshold = float(support_threshold)
        self.zero_low_support = bool(zero_low_support)
        self.out_dim = int(out_dim)
        if self.sigma <= 0:
            raise ValueError("sigma must be positive.")
        self.phi = nn.Sequential(
            nn.Linear(latent_dim + 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def _decode_chunk(self, X_chunk: Tensor, h_flat: Tensor, P_flat: Tensor) -> Tensor:
        B, M, Hdim = h_flat.shape
        N_chunk = X_chunk.shape[0]

        dr = X_chunk[None, :, None, :] - P_flat[None, None, :, :]  # [1, Nc, M, 2]
        dist2 = torch.sum(dr * dr, dim=-1)
        w = torch.exp(-dist2 / (self.sigma**2))
        support = w.sum(dim=-1, keepdim=True)
        if self.normalize:
            w = w / support.clamp_min(self.support_eps)
        if self.zero_low_support and self.support_threshold > 0.0:
            w = torch.where(support > self.support_threshold, w, torch.zeros_like(w))

        h_expand = h_flat[:, None, :, :].expand(B, N_chunk, M, Hdim)
        dr_expand = dr.expand(B, N_chunk, M, 2)
        msg = self.phi(torch.cat([h_expand, dr_expand], dim=-1))
        return torch.sum(w[..., None] * msg, dim=2)  # [B, Nc, out_dim]

    def forward(self, X: Tensor, h: Tensor, P: Tensor) -> Tensor:
        """Forward pass.

        Args:
            X: normalized active-node coordinates, [N, 2].
            h: latent features, [B, NLy, NLx, latent_dim].
            P: latent grid coordinates, [NLy, NLx, 2].

        Returns:
            nodal output, [B, N, out_dim].
        """
        if X.ndim != 2 or X.shape[-1] != 2:
            raise ValueError("X must have shape [N, 2].")
        if h.ndim != 4:
            raise ValueError("h must have shape [B, NLy, NLx, C].")
        if P.ndim != 3 or P.shape[-1] != 2:
            raise ValueError("P must have shape [NLy, NLx, 2].")
        if tuple(h.shape[1:3]) != tuple(P.shape[:2]):
            raise ValueError(f"h latent grid {tuple(h.shape[1:3])} != P grid {tuple(P.shape[:2])}.")

        X = X.to(device=P.device, dtype=P.dtype)
        h = h.to(device=P.device, dtype=P.dtype)
        P_flat = P.reshape(-1, 2)
        h_flat = h.reshape(h.shape[0], -1, h.shape[-1])

        if self.chunk_size is None:
            return self._decode_chunk(X, h_flat, P_flat)

        chunks = []
        for start in range(0, X.shape[0], self.chunk_size):
            chunks.append(self._decode_chunk(X[start : start + self.chunk_size], h_flat, P_flat))
        return torch.cat(chunks, dim=1)


# -----------------------------------------------------------------------------
# Final 2D model
# -----------------------------------------------------------------------------

class GAMFNO2D(nn.Module):
    """Reusable 2D Geometry-Adaptive/Geometry-Aware FNO model.

    Args:
        node_in_dim: number of active-node feature channels.
        out_dim: output field dimension, usually 2 for [u, v].
        backbone: "fno", "cnn", or "mlp".
        latent_dim: latent channel dimension.
        n_latent: tuple/list/int for latent grid size. If int, uses square grid.
        radial_hidden: hidden width of radial encoder/decoder MLPs.
        sigma_enc/sigma_dec: radial kernel bandwidths in normalized coordinates.
        width: latent backbone width.
        modes1/modes2: Fourier modes for FNO backbone.
        layers: number of latent backbone layers/channels.
        fc_dim: projection/channel hidden width.
        fno_padding: optional right/bottom padding inside the FNO latent backbone.
        append_latent_coords: concatenate latent [x, y] coordinates before the
            backbone and project back to latent_dim.
        bc_mode: "scatter" or "none".
        freeze_radial: if True, radial encoder/decoder parameters are frozen.
        encoder_chunk_size / decoder_chunk_size: optional chunk sizes to reduce
            memory in radial aggregation for large active node or latent sets.
        mask_latent: if True, zero latent nodes with low Gaussian support before
            and after the latent backbone.
        latent_support_threshold: threshold on unnormalized radial support used
            to identify empty latent nodes in holes/notches/outside domains. The
            default is conservative and should be tuned with sigma_enc and the
            active-node spacing.
        output_scale/output_shift: optional per-output affine transform applied
            to raw decoder outputs before BC enforcement. Use this for
            nondimensionalized multiphysics fields.
    """

    def __init__(
        self,
        node_in_dim: int,
        out_dim: int = 2,
        backbone: str = "fno",
        latent_dim: int = 32,
        n_latent: Union[int, Sequence[int]] = (64, 64),
        radial_hidden: int = 64,
        sigma_enc: float = 0.08,
        sigma_dec: float = 0.08,
        width: int = 64,
        modes1: Union[int, Sequence[int]] = 8,
        modes2: Union[int, Sequence[int]] = 8,
        layers: int = 4,
        fc_dim: int = 128,
        fno_padding: Union[int, Sequence[int]] = 0,
        append_latent_coords: bool = True,
        bc_mode: str = "scatter",
        freeze_radial: bool = False,
        encoder_chunk_size: Optional[int] = None,
        decoder_chunk_size: Optional[int] = None,
        mask_latent: bool = True,
        latent_support_threshold: float = 1e-10,
        radial_support_eps: float = 1e-12,
        output_scale: Union[float, Sequence[float], Tensor] = 1.0,
        output_shift: Union[float, Sequence[float], Tensor] = 0.0,
    ):
        super().__init__()
        self.backbone_name = backbone.lower()
        self.bc_mode = bc_mode.lower()
        self.out_dim = int(out_dim)
        self.latent_dim = int(latent_dim)
        self.n_latent = _pair(n_latent)
        self.append_latent_coords = bool(append_latent_coords)
        self.mask_latent = bool(mask_latent)
        self.latent_support_threshold = float(latent_support_threshold)

        P = make_latent_grid(self.n_latent, dtype=torch.float32)
        self.register_buffer("P", P)
        self.register_buffer("output_scale", _as_channel_buffer(output_scale, self.out_dim))
        self.register_buffer("output_shift", _as_channel_buffer(output_shift, self.out_dim))

        self.encoder = RadialEncoder2D(
            node_in_dim=node_in_dim,
            latent_dim=latent_dim,
            hidden=radial_hidden,
            sigma=sigma_enc,
            normalize=True,
            chunk_size=encoder_chunk_size,
            support_eps=radial_support_eps,
            support_threshold=latent_support_threshold,
            zero_low_support=mask_latent,
        )
        self.bulk_lift = (
            nn.Linear(latent_dim + 2, latent_dim)
            if self.append_latent_coords
            else nn.Identity()
        )
        self.bulk = make_latent_backbone_2d(
            backbone=self.backbone_name,
            latent_dim=latent_dim,
            n_latent=self.n_latent,
            width=width,
            modes1=modes1,
            modes2=modes2,
            layers=layers,
            fc_dim=fc_dim,
            fno_padding=fno_padding,
        )
        self.decoder = RadialDecoder2D(
            latent_dim=latent_dim,
            out_dim=out_dim,
            hidden=radial_hidden,
            sigma=sigma_dec,
            normalize=True,
            chunk_size=decoder_chunk_size,
            support_eps=radial_support_eps,
        )

        if freeze_radial:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
            for p in self.decoder.parameters():
                p.requires_grad_(False)

    def _apply_output_transform(self, raw: Tensor) -> Tensor:
        scale = self.output_scale.to(device=raw.device, dtype=raw.dtype)
        shift = self.output_shift.to(device=raw.device, dtype=raw.dtype)
        return raw * scale.view(1, 1, -1) + shift.view(1, 1, -1)

    def _latent_mask_float(self, X: Tensor, batch_size: int, dtype: torch.dtype) -> Optional[Tensor]:
        if not self.mask_latent:
            return None
        mask = self.encoder.support_mask(X, self.P, threshold=self.latent_support_threshold)
        if not bool(mask.any()):
            raise ValueError(
                "All latent nodes are below latent_support_threshold. Increase sigma_enc or lower latent_support_threshold."
            )
        return mask.to(device=self.P.device, dtype=dtype).unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1, -1)

    @staticmethod
    def _expand_output_mask(output_mask: Tensor, target: Tensor) -> Tensor:
        mask = output_mask.to(device=target.device, dtype=torch.bool)
        if mask.ndim == 1:
            mask = mask[None, :, None]
        elif mask.ndim == 2:
            mask = mask[None, :, :]
        elif mask.ndim != 3:
            raise ValueError("output_mask must have shape [N], [N, D], or [B, N, D].")
        return torch.broadcast_to(mask, target.shape)

    def _assert_bc_not_masked_out(
        self,
        raw: Tensor,
        bc_mask: Optional[Tensor],
        output_mask: Optional[Tensor],
    ) -> None:
        if bc_mask is None or output_mask is None:
            return
        bc = bc_mask.to(device=raw.device, dtype=torch.bool)
        if bc.ndim == 2:
            bc = bc[None, :, :]
        elif bc.ndim != 3:
            raise ValueError("bc_mask must have shape [N, D] or [B, N, D].")
        bc = torch.broadcast_to(bc, raw.shape)
        out = self._expand_output_mask(output_mask, raw)
        if bool((bc & (~out)).any()):
            raise ValueError(
                "A Dirichlet-constrained node/channel is inactive in output_mask. "
                "This would zero the prescribed value after hard_bc. Keep constrained nodes active."
            )

    def hard_bc(
        self,
        raw: Tensor,
        bc_mask: Optional[Tensor] = None,
        bc_val: Optional[Tensor] = None,
        bc_mode: Optional[str] = None,
    ) -> Tensor:
        mode = self.bc_mode if bc_mode is None else bc_mode.lower()
        if mode == "scatter":
            return hard_scatter_bc(raw, bc_mask, bc_val)
        if mode == "none":
            return raw
        raise ValueError("bc_mode must be one of: 'scatter', 'none'.")

    def apply_output_mask(self, u: Tensor, output_mask: Optional[Tensor]) -> Tensor:
        """Zero out inactive output nodes, if output_mask is provided."""
        if output_mask is None:
            return u
        mask = self._expand_output_mask(output_mask, u)
        return torch.where(mask, u, torch.zeros_like(u))

    def forward(
        self,
        X: Tensor,
        c: Tensor,
        bc_mask: Optional[Tensor] = None,
        bc_val: Optional[Tensor] = None,
        output_mask: Optional[Tensor] = None,
        bc_mode: Optional[str] = None,
        return_dict: bool = False,
    ):
        """Run the model.

        Args:
            X: normalized active-node coordinates, [N, 2].
            c: active-node features, [N, C] or [B, N, C]. Batch means same X.
            bc_mask: optional boolean mask, [N, out_dim] or [B, N, out_dim].
            bc_val: optional Dirichlet values, same/broadcastable shape as bc_mask.
            output_mask: optional active output mask. If [N], applies to all output
                channels; if [N, out_dim], applies per channel.
            bc_mode: optional override of self.bc_mode.
            return_dict: if True, return latent tensors and raw/scattered outputs.

        Returns:
            u if return_dict=False. Shape [N, out_dim] for unbatched input or
            [B, N, out_dim] for batched input.
        """
        unbatched = c.ndim == 2
        X = X.to(device=self.P.device, dtype=self.P.dtype)
        c = c.to(device=self.P.device, dtype=self.P.dtype)

        h0 = self.encoder(X, c, self.P)  # [B, NLy, NLx, latent_dim]
        latent_mask = self._latent_mask_float(X, h0.shape[0], h0.dtype)
        if latent_mask is not None:
            h0 = h0 * latent_mask

        if self.append_latent_coords:
            P_batched = self.P.unsqueeze(0).expand(h0.shape[0], -1, -1, -1)
            h_in = self.bulk_lift(torch.cat([h0, P_batched], dim=-1))
        else:
            h_in = self.bulk_lift(h0)
        if latent_mask is not None:
            h_in = h_in * latent_mask

        h = self.bulk(h_in)
        if latent_mask is not None:
            h = h * latent_mask

        raw = self.decoder(X, h, self.P)  # [B, N, out_dim]
        raw = self._apply_output_transform(raw)
        self._assert_bc_not_masked_out(raw, bc_mask, output_mask)
        u = self.hard_bc(raw, bc_mask=bc_mask, bc_val=bc_val, bc_mode=bc_mode)
        u = self.apply_output_mask(u, output_mask)

        if unbatched:
            u_out = u[0]
            raw_out = raw[0]
        else:
            u_out = u
            raw_out = raw

        if return_dict:
            return {
                "latent_in": h0,
                "latent_after_lift": h_in,
                "latent": h,
                "latent_mask": None if latent_mask is None else latent_mask[0, :, :, 0].bool(),
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
        model: GAMFNO2D,
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


# Backward-compatible alias if a notebook uses GAMNO2D naming.
GAMNO2D = GAMFNO2D


__all__ = [
    "GAMFNO2D",
    "GAMNO2D",
    "StatefulIncrementalWrapper",
    "RadialEncoder2D",
    "RadialDecoder2D",
    "FNO2d",
    "SpectralConv2d",
    "CNN2dBulk",
    "MLP2dBulk",
    "make_latent_backbone_2d",
    "make_latent_grid",
    "hard_scatter_bc",
    "count_trainable_params",
]
