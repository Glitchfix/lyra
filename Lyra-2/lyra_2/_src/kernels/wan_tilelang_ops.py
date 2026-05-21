# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional TileLang kernels for WAN/Lyra inference glue ops."""

from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import tilelang
    import tilelang.language as T

    TILELANG_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only on systems without TileLang.
    tilelang = None
    T = None
    TILELANG_AVAILABLE = False


_STATS = {
    "gelu_calls": 0,
    "gelu_elements": 0,
    "gelu_fallback_calls": 0,
    "gelu_fallback_elements": 0,
    "rmsnorm_calls": 0,
    "rmsnorm_shared_input_calls": 0,
    "rmsnorm_elements": 0,
    "layernorm_mod_calls": 0,
    "layernorm_mod_elements": 0,
    "gated_residual_calls": 0,
    "gated_residual_elements": 0,
}


def reset_tilelang_wan_stats() -> None:
    """Reset runtime hit counters for optional WAN TileLang kernels."""
    for key in _STATS:
        _STATS[key] = 0


def get_tilelang_wan_stats() -> dict[str, int]:
    """Return runtime hit counters for optional WAN TileLang kernels."""
    return dict(_STATS)


def _dtype_to_tilelang(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float32:
        return "float32"
    raise TypeError(f"TileLang WAN kernels do not support dtype={dtype}.")


def can_use_tilelang_gelu(x: torch.Tensor) -> bool:
    """Return whether the in-place GELU kernel supports this tensor."""
    return (
        TILELANG_AVAILABLE
        and torch.cuda.is_available()
        and x.is_cuda
        and x.is_contiguous()
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and x.numel() > 0
    )


def can_use_tilelang_rmsnorm(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """Return whether the row-wise RMSNorm kernel supports these inputs."""
    return (
        TILELANG_AVAILABLE
        and torch.cuda.is_available()
        and x.is_cuda
        and weight.is_cuda
        and x.is_contiguous()
        and weight.is_contiguous()
        and x.ndim >= 2
        and x.shape[-1] == weight.shape[0]
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and weight.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and x.numel() > 0
    )


def can_use_tilelang_layernorm_mod(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> bool:
    """Return whether fused LayerNorm + timestep modulation supports these inputs."""
    return (
        TILELANG_AVAILABLE
        and torch.cuda.is_available()
        and x.is_cuda
        and scale.is_cuda
        and shift.is_cuda
        and x.is_contiguous()
        and x.ndim == 3
        and scale.shape[0] == x.shape[0]
        and shift.shape[0] == x.shape[0]
        and scale.shape[-1] == x.shape[-1]
        and shift.shape[-1] == x.shape[-1]
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and scale.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and shift.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and x.numel() > 0
    )


def can_use_tilelang_gated_residual(x: torch.Tensor, y: torch.Tensor, gate: torch.Tensor) -> bool:
    """Return whether fused ``x += y * gate`` supports these inputs."""
    return (
        TILELANG_AVAILABLE
        and torch.cuda.is_available()
        and x.is_cuda
        and y.is_cuda
        and gate.is_cuda
        and x.is_contiguous()
        and y.is_contiguous()
        and x.shape == y.shape
        and x.ndim == 3
        and gate.shape[0] == x.shape[0]
        and gate.shape[-1] == x.shape[-1]
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and y.dtype == x.dtype
        and gate.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and x.numel() > 0
    )


if TILELANG_AVAILABLE:

    @tilelang.jit(target="cuda")
    def _gelu_tanh_inplace_kernel(numel: int, dtype: str, block_size: int = 256):
        coeff = 0.7978845608028654
        cubic = 0.044715

        @T.prim_func
        def kernel(x: T.Tensor((numel,), dtype)):
            with T.Kernel(T.ceildiv(numel, block_size), threads=block_size) as bx:
                for i in T.Parallel(block_size):
                    off = bx * block_size + i
                    if off < numel:
                        v = T.cast(x[off], "float32")
                        v3 = v * v * v
                        y = 0.5 * v * (1.0 + T.tanh(coeff * (v + cubic * v3)))
                        x[off] = T.cast(y, dtype)

        return kernel

    @tilelang.jit(target="cuda")
    def _rmsnorm_kernel(rows: int, cols: int, dtype: str, weight_dtype: str, threads: int = 256):
        @T.prim_func
        def kernel(
            x: T.Tensor((rows, cols), dtype),
            weight: T.Tensor((cols,), weight_dtype),
            out: T.Tensor((rows, cols), dtype),
            eps: T.float32,
        ):
            with T.Kernel(rows, threads=threads) as row:
                partial = T.alloc_shared((threads,), "float32")
                sum_buf = T.alloc_fragment((1,), "float32")

                for tx in T.Parallel(threads):
                    acc = T.alloc_var("float32", init=0.0)
                    for j in T.serial(0, T.ceildiv(cols, threads)):
                        col = j * threads + tx
                        if col < cols:
                            v = T.cast(x[row, col], "float32")
                            acc = acc + v * v
                    partial[tx] = acc

                T.reduce_sum(partial, sum_buf, dim=0)
                inv_rms = T.rsqrt(sum_buf[0] / T.cast(cols, "float32") + eps)

                for tx in T.Parallel(threads):
                    for j in T.serial(0, T.ceildiv(cols, threads)):
                        col = j * threads + tx
                        if col < cols:
                            v = T.cast(x[row, col], "float32")
                            w = T.cast(weight[col], "float32")
                            normed = T.cast(v * inv_rms, dtype)
                            out[row, col] = T.cast(T.cast(normed, "float32") * w, dtype)

        return kernel

    @tilelang.jit(target="cuda")
    def _rmsnorm_shared_input_kernel(rows: int, cols: int, dtype: str, weight_dtype: str, threads: int = 256):
        @T.prim_func
        def kernel(
            x: T.Tensor((rows, cols), dtype),
            weight: T.Tensor((cols,), weight_dtype),
            out: T.Tensor((rows, cols), dtype),
            eps: T.float32,
        ):
            with T.Kernel(rows, threads=threads) as row:
                partial = T.alloc_shared((threads,), "float32")
                x_cache = T.alloc_shared((cols,), "float32")
                sum_buf = T.alloc_fragment((1,), "float32")

                for tx in T.Parallel(threads):
                    acc = T.alloc_var("float32", init=0.0)
                    for j in T.serial(0, T.ceildiv(cols, threads)):
                        col = j * threads + tx
                        if col < cols:
                            v = T.cast(x[row, col], "float32")
                            x_cache[col] = v
                            acc = acc + v * v
                    partial[tx] = acc

                T.reduce_sum(partial, sum_buf, dim=0)
                inv_rms = T.rsqrt(sum_buf[0] / T.cast(cols, "float32") + eps)

                for tx in T.Parallel(threads):
                    for j in T.serial(0, T.ceildiv(cols, threads)):
                        col = j * threads + tx
                        if col < cols:
                            v = x_cache[col]
                            w = T.cast(weight[col], "float32")
                            normed = T.cast(v * inv_rms, dtype)
                            out[row, col] = T.cast(T.cast(normed, "float32") * w, dtype)

        return kernel

    @tilelang.jit(target="cuda")
    def _layernorm_mod_kernel(
        batch: int,
        rows_per_batch: int,
        cols: int,
        dtype: str,
        mod_dtype: str,
        threads: int = 256,
    ):
        rows = batch * rows_per_batch

        @T.prim_func
        def kernel(
            x: T.Tensor((batch, rows_per_batch, cols), dtype),
            scale_delta: T.Tensor((batch, cols), mod_dtype),
            shift: T.Tensor((batch, cols), mod_dtype),
            out: T.Tensor((batch, rows_per_batch, cols), dtype),
            eps: T.float32,
        ):
            with T.Kernel(rows, threads=threads) as row:
                partial_sum = T.alloc_shared((threads,), "float32")
                partial_sq = T.alloc_shared((threads,), "float32")
                sum_buf = T.alloc_fragment((1,), "float32")
                sq_buf = T.alloc_fragment((1,), "float32")
                b = row // rows_per_batch
                r = row - b * rows_per_batch

                for tx in T.Parallel(threads):
                    acc = T.alloc_var("float32", init=0.0)
                    acc_sq = T.alloc_var("float32", init=0.0)
                    for j in T.serial(0, T.ceildiv(cols, threads)):
                        col = j * threads + tx
                        if col < cols:
                            v = T.cast(x[b, r, col], "float32")
                            acc = acc + v
                            acc_sq = acc_sq + v * v
                    partial_sum[tx] = acc
                    partial_sq[tx] = acc_sq

                T.reduce_sum(partial_sum, sum_buf, dim=0)
                T.reduce_sum(partial_sq, sq_buf, dim=0)
                mean = sum_buf[0] / T.cast(cols, "float32")
                var = sq_buf[0] / T.cast(cols, "float32") - mean * mean
                inv_std = T.rsqrt(T.max(var, 0.0) + eps)

                for tx in T.Parallel(threads):
                    for j in T.serial(0, T.ceildiv(cols, threads)):
                        col = j * threads + tx
                        if col < cols:
                            v = T.cast(x[b, r, col], "float32")
                            scale = 1.0 + T.cast(scale_delta[b, col], "float32")
                            bias = T.cast(shift[b, col], "float32")
                            normed = T.cast((v - mean) * inv_std, dtype)
                            y = T.cast(normed, "float32") * scale + bias
                            out[b, r, col] = T.cast(y, dtype)

        return kernel

    @tilelang.jit(target="cuda")
    def _gated_residual_add_kernel(
        batch: int,
        rows_per_batch: int,
        cols: int,
        dtype: str,
        gate_dtype: str,
        block_size: int = 256,
    ):
        numel = batch * rows_per_batch * cols

        @T.prim_func
        def kernel(
            x: T.Tensor((batch, rows_per_batch, cols), dtype),
            y: T.Tensor((batch, rows_per_batch, cols), dtype),
            gate: T.Tensor((batch, cols), gate_dtype),
        ):
            with T.Kernel(T.ceildiv(numel, block_size), threads=block_size) as bx:
                for tx in T.Parallel(block_size):
                    off = bx * block_size + tx
                    if off < numel:
                        c = off % cols
                        row = off // cols
                        b = row // rows_per_batch
                        r = row - b * rows_per_batch
                        xv = T.cast(x[b, r, c], "float32")
                        yv = T.cast(y[b, r, c], "float32")
                        gv = T.cast(T.cast(gate[b, c], dtype), "float32")
                        scaled = T.cast(yv * gv, dtype)
                        x[b, r, c] = T.cast(xv + T.cast(scaled, "float32"), dtype)

        return kernel


@lru_cache(maxsize=32)
def _compiled_gelu_tanh_inplace(numel: int, dtype: str, block_size: int):
    if not TILELANG_AVAILABLE:
        raise RuntimeError("TileLang is not available.")
    return _gelu_tanh_inplace_kernel(numel, dtype, block_size)


@lru_cache(maxsize=32)
def _compiled_rmsnorm(rows: int, cols: int, dtype: str, weight_dtype: str, threads: int):
    if not TILELANG_AVAILABLE:
        raise RuntimeError("TileLang is not available.")
    return _rmsnorm_kernel(rows, cols, dtype, weight_dtype, threads)


@lru_cache(maxsize=32)
def _compiled_rmsnorm_shared_input(rows: int, cols: int, dtype: str, weight_dtype: str, threads: int):
    if not TILELANG_AVAILABLE:
        raise RuntimeError("TileLang is not available.")
    return _rmsnorm_shared_input_kernel(rows, cols, dtype, weight_dtype, threads)


@lru_cache(maxsize=32)
def _compiled_layernorm_mod(batch: int, rows_per_batch: int, cols: int, dtype: str, mod_dtype: str, threads: int):
    if not TILELANG_AVAILABLE:
        raise RuntimeError("TileLang is not available.")
    return _layernorm_mod_kernel(batch, rows_per_batch, cols, dtype, mod_dtype, threads)


@lru_cache(maxsize=32)
def _compiled_gated_residual_add(
    batch: int,
    rows_per_batch: int,
    cols: int,
    dtype: str,
    gate_dtype: str,
    block_size: int,
):
    if not TILELANG_AVAILABLE:
        raise RuntimeError("TileLang is not available.")
    return _gated_residual_add_kernel(batch, rows_per_batch, cols, dtype, gate_dtype, block_size)


def _prefer_shared_input_rmsnorm(rows: int, cols: int, dtype: torch.dtype, threads: int) -> bool:
    bytes_per_block = int(cols) * 4 + int(threads) * 4
    return int(rows) <= 1024 and dtype in (torch.float16, torch.bfloat16) and int(cols) <= 4096 and bytes_per_block <= 17_408


def gelu_tanh_inplace_tilelang(x: torch.Tensor, *, block_size: int = 256) -> torch.Tensor:
    """Apply tanh-approx GELU in place and return ``x``."""
    if not can_use_tilelang_gelu(x):
        raise RuntimeError("TileLang GELU kernel does not support this tensor.")

    flat = x.reshape(-1)
    kernel = _compiled_gelu_tanh_inplace(
        int(flat.numel()),
        _dtype_to_tilelang(flat.dtype),
        int(block_size),
    )
    kernel(flat)
    _STATS["gelu_calls"] += 1
    _STATS["gelu_elements"] += int(flat.numel())
    return x


def rmsnorm_tilelang(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    threads: int = 256,
    shared_input: bool | None = None,
) -> torch.Tensor:
    """Apply row-wise RMSNorm over the last dimension."""
    if not can_use_tilelang_rmsnorm(x, weight):
        raise RuntimeError("TileLang RMSNorm kernel does not support these inputs.")

    cols = int(x.shape[-1])
    flat = x.reshape(-1, cols)
    out = torch.empty_like(flat)
    rows = int(flat.shape[0])
    use_shared_input = (
        _prefer_shared_input_rmsnorm(rows, cols, flat.dtype, int(threads))
        if shared_input is None
        else bool(shared_input)
    )
    if use_shared_input:
        kernel = _compiled_rmsnorm_shared_input(
            rows,
            cols,
            _dtype_to_tilelang(flat.dtype),
            _dtype_to_tilelang(weight.dtype),
            int(threads),
        )
    else:
        kernel = _compiled_rmsnorm(
            rows,
            cols,
            _dtype_to_tilelang(flat.dtype),
            _dtype_to_tilelang(weight.dtype),
            int(threads),
        )
    kernel(flat, weight, out, float(eps))
    _STATS["rmsnorm_calls"] += 1
    if use_shared_input:
        _STATS["rmsnorm_shared_input_calls"] += 1
    _STATS["rmsnorm_elements"] += int(flat.numel())
    return out.reshape_as(x)


def layernorm_mod_tilelang(
    x: torch.Tensor,
    scale_delta: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
    *,
    threads: int = 256,
) -> torch.Tensor:
    """Apply LayerNorm(x) * (1 + scale_delta) + shift over the last dimension."""
    if not can_use_tilelang_layernorm_mod(x, scale_delta, shift):
        raise RuntimeError("TileLang modulated LayerNorm kernel does not support these inputs.")

    B, L, C = (int(x.shape[0]), int(x.shape[1]), int(x.shape[2]))
    scale = scale_delta.reshape(B, C).contiguous()
    bias = shift.reshape(B, C).contiguous()
    out = torch.empty_like(x)
    kernel = _compiled_layernorm_mod(
        B,
        L,
        C,
        _dtype_to_tilelang(x.dtype),
        _dtype_to_tilelang(scale.dtype),
        int(threads),
    )
    kernel(x, scale, bias, out, float(eps))
    _STATS["layernorm_mod_calls"] += 1
    _STATS["layernorm_mod_elements"] += int(x.numel())
    return out


def gated_residual_add_tilelang(
    x: torch.Tensor,
    y: torch.Tensor,
    gate: torch.Tensor,
    *,
    block_size: int = 256,
) -> torch.Tensor:
    """Apply ``x += y * gate`` in place and return ``x``."""
    if not can_use_tilelang_gated_residual(x, y, gate):
        raise RuntimeError("TileLang gated residual kernel does not support these inputs.")

    B, L, C = (int(x.shape[0]), int(x.shape[1]), int(x.shape[2]))
    gate_flat = gate.reshape(B, C).contiguous()
    kernel = _compiled_gated_residual_add(
        B,
        L,
        C,
        _dtype_to_tilelang(x.dtype),
        _dtype_to_tilelang(gate_flat.dtype),
        int(block_size),
    )
    kernel(x, y, gate_flat)
    _STATS["gated_residual_calls"] += 1
    _STATS["gated_residual_elements"] += int(x.numel())
    return x


class TileLangGELU(nn.Module):
    """Inference-time in-place tanh GELU with a safe PyTorch fallback."""

    def __init__(self, approximate: str = "tanh", min_elements: int = 16_000_000, block_size: int = 512) -> None:
        super().__init__()
        self.approximate = str(approximate)
        self.min_elements = int(min_elements)
        self.block_size = int(block_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            self.approximate != "tanh"
            or torch.is_grad_enabled()
            or x.numel() < self.min_elements
            or not can_use_tilelang_gelu(x)
        ):
            _STATS["gelu_fallback_calls"] += 1
            _STATS["gelu_fallback_elements"] += int(x.numel())
            return F.gelu(x, approximate=self.approximate)
        try:
            return gelu_tanh_inplace_tilelang(x, block_size=self.block_size)
        except Exception:
            _STATS["gelu_fallback_calls"] += 1
            _STATS["gelu_fallback_elements"] += int(x.numel())
            return F.gelu(x, approximate=self.approximate)


def replace_gelu_with_tilelang(module: nn.Module, *, min_elements: int = 16_000_000, block_size: int = 512) -> int:
    """Replace tanh GELU modules in-place and return the replacement count."""
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.GELU) and getattr(child, "approximate", "none") == "tanh":
            setattr(module, name, TileLangGELU(approximate="tanh", min_elements=min_elements, block_size=block_size))
            count += 1
        else:
            count += replace_gelu_with_tilelang(child, min_elements=min_elements, block_size=block_size)
    return count
