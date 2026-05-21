# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark and validate optional TileLang WAN inference ops."""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from lyra_2._src.kernels.wan_tilelang_ops import (
    TILELANG_AVAILABLE,
    gelu_tanh_inplace_tilelang,
    gated_residual_add_tilelang,
    layernorm_mod_tilelang,
    rmsnorm_tilelang,
)

try:
    import transformer_engine.pytorch as te
    from transformer_engine.pytorch.attention import DotProductAttention

    TRANSFORMER_ENGINE_AVAILABLE = True
except Exception:
    te = None
    DotProductAttention = None
    TRANSFORMER_ENGINE_AVAILABLE = False


def _time_cuda(fn, iterations: int) -> float:
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / float(iterations)


def _peak_alloc_mib(fn) -> float:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return float(max(0, peak - baseline)) / 1024.0 / 1024.0


def _copy_linear(src: nn.Linear, dst: nn.Module) -> None:
    with torch.no_grad():
        dst.weight.copy_(src.weight)
        if src.bias is not None and getattr(dst, "bias", None) is not None:
            dst.bias.copy_(src.bias)


def _dtype(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _run_gelu(args: argparse.Namespace, dtype: torch.dtype) -> None:
    x = torch.randn(args.gelu_rows, args.hidden_dim, device="cuda", dtype=dtype)

    ref = F.gelu(x, approximate="tanh")
    tile_in = x.clone()
    tile = gelu_tanh_inplace_tilelang(tile_in, block_size=args.gelu_block_size)
    torch.cuda.synchronize()
    diff = (ref - tile).abs().float()

    torch_ms = _time_cuda(lambda: F.gelu(x, approximate="tanh"), args.iterations)
    tile_input = x.clone()
    tile_ms = _time_cuda(lambda: gelu_tanh_inplace_tilelang(tile_input, block_size=args.gelu_block_size), args.iterations)

    torch_peak = _peak_alloc_mib(lambda: F.gelu(x, approximate="tanh"))
    tile_peak = _peak_alloc_mib(lambda: gelu_tanh_inplace_tilelang(tile_input, block_size=args.gelu_block_size))

    print(f"gelu_shape: rows={args.gelu_rows}, cols={args.hidden_dim}, dtype={args.dtype}")
    print(f"gelu_max_abs_diff: {float(diff.max().item()):.6g}")
    print(f"gelu_mean_abs_diff: {float(diff.mean().item()):.6g}")
    print(f"gelu_torch_ms: {torch_ms:.4f}")
    print(f"gelu_tilelang_ms: {tile_ms:.4f}")
    print(f"gelu_tilelang_speedup: {torch_ms / max(tile_ms, 1.0e-9):.2f}x")
    print(f"gelu_torch_peak_mib: {torch_peak:.3f}")
    print(f"gelu_tilelang_peak_mib: {tile_peak:.3f}")


def _run_rmsnorm(args: argparse.Namespace, dtype: torch.dtype) -> None:
    x = torch.randn(args.rms_rows, args.model_dim, device="cuda", dtype=dtype)
    weight = torch.randn(args.model_dim, device="cuda", dtype=dtype)

    def torch_rmsnorm() -> torch.Tensor:
        normed = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + args.eps)
        return normed.to(dtype) * weight

    ref = torch_rmsnorm()
    shared_input = None if args.rms_shared_input == "auto" else args.rms_shared_input == "on"
    tile = rmsnorm_tilelang(x, weight, args.eps, threads=args.rms_threads, shared_input=shared_input)
    torch.cuda.synchronize()
    diff = (ref - tile).abs().float()

    torch_ms = _time_cuda(torch_rmsnorm, args.iterations)
    tile_ms = _time_cuda(
        lambda: rmsnorm_tilelang(x, weight, args.eps, threads=args.rms_threads, shared_input=shared_input),
        args.iterations,
    )
    torch_peak = _peak_alloc_mib(torch_rmsnorm)
    tile_peak = _peak_alloc_mib(
        lambda: rmsnorm_tilelang(x, weight, args.eps, threads=args.rms_threads, shared_input=shared_input)
    )

    print(f"rms_shape: rows={args.rms_rows}, cols={args.model_dim}, dtype={args.dtype}")
    print(f"rms_max_abs_diff: {float(diff.max().item()):.6g}")
    print(f"rms_mean_abs_diff: {float(diff.mean().item()):.6g}")
    print(f"rms_torch_ms: {torch_ms:.4f}")
    print(f"rms_tilelang_ms: {tile_ms:.4f}")
    print(f"rms_tilelang_speedup: {torch_ms / max(tile_ms, 1.0e-9):.2f}x")
    print(f"rms_torch_peak_mib: {torch_peak:.3f}")
    print(f"rms_tilelang_peak_mib: {tile_peak:.3f}")


def _run_layernorm_mod(args: argparse.Namespace, dtype: torch.dtype) -> None:
    x = torch.randn(1, args.mod_rows, args.model_dim, device="cuda", dtype=dtype)
    scale_delta = torch.randn(1, 1, args.model_dim, device="cuda", dtype=torch.float32)
    shift = torch.randn(1, 1, args.model_dim, device="cuda", dtype=torch.float32)

    def torch_layernorm_mod() -> torch.Tensor:
        out = F.layer_norm(x, (args.model_dim,), eps=args.eps).float()
        return (out * (1.0 + scale_delta) + shift).type_as(x)

    ref = torch_layernorm_mod()
    tile = layernorm_mod_tilelang(x, scale_delta, shift, args.eps, threads=args.mod_threads)
    torch.cuda.synchronize()
    diff = (ref - tile).abs().float()

    torch_ms = _time_cuda(torch_layernorm_mod, args.iterations)
    tile_ms = _time_cuda(
        lambda: layernorm_mod_tilelang(x, scale_delta, shift, args.eps, threads=args.mod_threads),
        args.iterations,
    )
    torch_peak = _peak_alloc_mib(torch_layernorm_mod)
    tile_peak = _peak_alloc_mib(
        lambda: layernorm_mod_tilelang(x, scale_delta, shift, args.eps, threads=args.mod_threads)
    )

    print(f"modln_shape: rows={args.mod_rows}, cols={args.model_dim}, dtype={args.dtype}")
    print(f"modln_max_abs_diff: {float(diff.max().item()):.6g}")
    print(f"modln_mean_abs_diff: {float(diff.mean().item()):.6g}")
    print(f"modln_torch_ms: {torch_ms:.4f}")
    print(f"modln_tilelang_ms: {tile_ms:.4f}")
    print(f"modln_tilelang_speedup: {torch_ms / max(tile_ms, 1.0e-9):.2f}x")
    print(f"modln_torch_peak_mib: {torch_peak:.3f}")
    print(f"modln_tilelang_peak_mib: {tile_peak:.3f}")


def _run_ffn(args: argparse.Namespace, dtype: torch.dtype) -> None:
    x = torch.randn(args.ffn_rows, args.model_dim, device="cuda", dtype=dtype)
    fc1 = nn.Linear(args.model_dim, args.hidden_dim, device="cuda", dtype=dtype)
    fc2 = nn.Linear(args.hidden_dim, args.model_dim, device="cuda", dtype=dtype)

    def torch_ffn() -> torch.Tensor:
        return fc2(F.gelu(fc1(x), approximate="tanh"))

    def tilelang_gelu_ffn() -> torch.Tensor:
        hidden = fc1(x)
        hidden = gelu_tanh_inplace_tilelang(hidden, block_size=args.gelu_block_size)
        return fc2(hidden)

    ref = torch_ffn()
    tile = tilelang_gelu_ffn()
    torch.cuda.synchronize()
    diff = (ref - tile).abs().float()

    torch_ms = _time_cuda(torch_ffn, args.iterations)
    tile_ms = _time_cuda(tilelang_gelu_ffn, args.iterations)
    torch_peak = _peak_alloc_mib(torch_ffn)
    tile_peak = _peak_alloc_mib(tilelang_gelu_ffn)

    print(f"ffn_shape: rows={args.ffn_rows}, in={args.model_dim}, hidden={args.hidden_dim}, dtype={args.dtype}")
    print(f"ffn_tile_gelu_max_abs_diff: {float(diff.max().item()):.6g}")
    print(f"ffn_tile_gelu_mean_abs_diff: {float(diff.mean().item()):.6g}")
    print(f"ffn_torch_ms: {torch_ms:.4f}")
    print(f"ffn_tile_gelu_ms: {tile_ms:.4f}")
    print(f"ffn_tile_gelu_speedup: {torch_ms / max(tile_ms, 1.0e-9):.2f}x")
    print(f"ffn_torch_peak_mib: {torch_peak:.3f}")
    print(f"ffn_tile_gelu_peak_mib: {tile_peak:.3f}")

    if not TRANSFORMER_ENGINE_AVAILABLE:
        print("ffn_te_linear: unavailable")
        return

    te_fc1 = te.Linear(args.model_dim, args.hidden_dim, bias=True, params_dtype=dtype).cuda()
    te_fc2 = te.Linear(args.hidden_dim, args.model_dim, bias=True, params_dtype=dtype).cuda()
    _copy_linear(fc1, te_fc1)
    _copy_linear(fc2, te_fc2)

    def te_ffn() -> torch.Tensor:
        return te_fc2(F.gelu(te_fc1(x), approximate="tanh"))

    te_out = te_ffn()
    torch.cuda.synchronize()
    te_diff = (ref - te_out).abs().float()
    te_ms = _time_cuda(te_ffn, args.iterations)
    te_peak = _peak_alloc_mib(te_ffn)
    print(f"ffn_te_max_abs_diff: {float(te_diff.max().item()):.6g}")
    print(f"ffn_te_mean_abs_diff: {float(te_diff.mean().item()):.6g}")
    print(f"ffn_te_ms: {te_ms:.4f}")
    print(f"ffn_te_speedup: {torch_ms / max(te_ms, 1.0e-9):.2f}x")
    print(f"ffn_te_peak_mib: {te_peak:.3f}")


def _run_gated_residual(args: argparse.Namespace, dtype: torch.dtype) -> None:
    x_base = torch.randn(1, args.residual_rows, args.model_dim, device="cuda", dtype=dtype)
    y_base = torch.randn_like(x_base)
    gate = torch.randn(1, 1, args.model_dim, device="cuda", dtype=torch.float32)

    def torch_gated_residual_ref() -> torch.Tensor:
        x = x_base.clone()
        y = y_base.clone()
        y.mul_(gate.type_as(y))
        x.add_(y)
        return x

    def tilelang_gated_residual_ref() -> torch.Tensor:
        x = x_base.clone()
        y = y_base.clone()
        return gated_residual_add_tilelang(x, y, gate, block_size=args.residual_block_size)

    ref = torch_gated_residual_ref()
    tile = tilelang_gated_residual_ref()
    torch.cuda.synchronize()
    diff = (ref - tile).abs().float()

    x_torch = x_base.clone()
    y_torch = y_base.clone()
    x_tile = x_base.clone()
    y_tile = y_base.clone()

    def torch_gated_residual_op() -> torch.Tensor:
        y_torch.mul_(gate.type_as(y_torch))
        x_torch.add_(y_torch)
        return x_torch

    def tilelang_gated_residual_op() -> torch.Tensor:
        return gated_residual_add_tilelang(x_tile, y_tile, gate, block_size=args.residual_block_size)

    torch_ms = _time_cuda(torch_gated_residual_op, args.iterations)
    tile_ms = _time_cuda(tilelang_gated_residual_op, args.iterations)
    torch_peak = _peak_alloc_mib(torch_gated_residual_op)
    tile_peak = _peak_alloc_mib(tilelang_gated_residual_op)

    print(f"residual_shape: rows={args.residual_rows}, cols={args.model_dim}, dtype={args.dtype}")
    print(f"residual_max_abs_diff: {float(diff.max().item()):.6g}")
    print(f"residual_mean_abs_diff: {float(diff.mean().item()):.6g}")
    print(f"residual_torch_ms: {torch_ms:.4f}")
    print(f"residual_tilelang_ms: {tile_ms:.4f}")
    print(f"residual_tilelang_speedup: {torch_ms / max(tile_ms, 1.0e-9):.2f}x")
    print(f"residual_torch_peak_mib: {torch_peak:.3f}")
    print(f"residual_tilelang_peak_mib: {tile_peak:.3f}")


def _run_attention(args: argparse.Namespace, dtype: torch.dtype) -> None:
    if not TRANSFORMER_ENGINE_AVAILABLE:
        print("attention_te: unavailable")
        return

    b = 1
    q = torch.randn(b, args.attn_seq_len, args.attn_heads, args.attn_head_dim, device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    attn = DotProductAttention(
        args.attn_heads,
        args.attn_head_dim,
        num_gqa_groups=args.attn_heads,
        attention_dropout=0,
        qkv_format="bshd",
        attn_mask_type="no_mask",
    )

    def te_attention() -> torch.Tensor:
        return attn(q, k, v)

    def torch_sdpa_attention() -> torch.Tensor:
        q_bhld = q.transpose(1, 2)
        k_bhld = k.transpose(1, 2)
        v_bhld = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q_bhld, k_bhld, v_bhld, dropout_p=0.0, is_causal=False)
        return out.transpose(1, 2)

    ref = te_attention()
    sdpa = torch_sdpa_attention()
    torch.cuda.synchronize()
    ref_flat = ref.reshape(b, args.attn_seq_len, -1)
    sdpa_flat = sdpa.reshape(b, args.attn_seq_len, -1)
    diff = (ref_flat - sdpa_flat).abs().float()

    te_ms = _time_cuda(te_attention, args.attn_iterations)
    sdpa_ms = _time_cuda(torch_sdpa_attention, args.attn_iterations)
    te_peak = _peak_alloc_mib(te_attention)
    sdpa_peak = _peak_alloc_mib(torch_sdpa_attention)

    print(
        f"attention_shape: B={b}, S={args.attn_seq_len}, heads={args.attn_heads}, "
        f"head_dim={args.attn_head_dim}, dtype={args.dtype}"
    )
    print(f"attention_sdpa_vs_te_max_abs_diff: {float(diff.max().item()):.6g}")
    print(f"attention_sdpa_vs_te_mean_abs_diff: {float(diff.mean().item()):.6g}")
    print(f"attention_te_ms: {te_ms:.4f}")
    print(f"attention_sdpa_ms: {sdpa_ms:.4f}")
    print(f"attention_sdpa_speedup_vs_te: {te_ms / max(sdpa_ms, 1.0e-9):.2f}x")
    print(f"attention_te_peak_mib: {te_peak:.3f}")
    print(f"attention_sdpa_peak_mib: {sdpa_peak:.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and benchmark optional TileLang WAN inference ops.")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="bf16")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--gelu_rows", type=int, default=8192)
    parser.add_argument("--hidden_dim", type=int, default=8192)
    parser.add_argument("--gelu_block_size", type=int, default=512)
    parser.add_argument("--ffn_rows", type=int, default=7812)
    parser.add_argument("--rms_rows", type=int, default=8192)
    parser.add_argument("--model_dim", type=int, default=2048)
    parser.add_argument("--rms_threads", type=int, default=256)
    parser.add_argument("--rms_shared_input", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--mod_rows", type=int, default=27440)
    parser.add_argument("--mod_threads", type=int, default=256)
    parser.add_argument("--residual_rows", type=int, default=27440)
    parser.add_argument("--residual_block_size", type=int, default=256)
    parser.add_argument("--attn_seq_len", type=int, default=4096)
    parser.add_argument("--attn_heads", type=int, default=16)
    parser.add_argument("--attn_head_dim", type=int, default=128)
    parser.add_argument("--attn_iterations", type=int, default=10)
    parser.add_argument("--eps", type=float, default=1.0e-6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")
    if not TILELANG_AVAILABLE:
        raise SystemExit("TileLang is not available.")

    dtype = _dtype(args.dtype)
    _run_gelu(args, dtype)
    _run_rmsnorm(args, dtype)
    _run_layernorm_mod(args, dtype)
    _run_ffn(args, dtype)
    _run_gated_residual(args, dtype)
    _run_attention(args, dtype)


if __name__ == "__main__":
    main()
