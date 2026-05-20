# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark and validate Sparse3DCache point-building kernels."""

from __future__ import annotations

import argparse
import time

import torch

from lyra_2._src.datasets.forward_warp_utils_pytorch import unproject_points
from lyra_2._src.kernels.sparse3d_cache_triton import depth_to_world_points_triton

try:
    from lyra_2._src.kernels.sparse3d_cache_tilelang import depth_to_world_points_tilelang
except Exception as exc:
    depth_to_world_points_tilelang = None
    _TILELANG_IMPORT_ERROR = exc
else:
    _TILELANG_IMPORT_ERROR = None


def _reference_depth_to_world(depth: torch.Tensor, w2c: torch.Tensor, K: torch.Tensor, downsample: int) -> torch.Tensor:
    depth_ds = depth[:, :, ::downsample, ::downsample]
    scale = 1.0 / float(downsample)
    K_scaled = K.clone()
    K_scaled[:, 0, 0] *= scale
    K_scaled[:, 1, 1] *= scale
    K_scaled[:, 0, 2] *= scale
    K_scaled[:, 1, 2] *= scale
    return unproject_points(
        depth=depth_ds,
        w2c=w2c,
        intrinsic=K_scaled,
        is_depth=True,
        is_ftheta=False,
        mask=(depth_ds > 0),
        return_sparse=False,
    )


def _make_inputs(batch: int, height: int, width: int, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device=device)
    gen.manual_seed(123)
    depth = torch.rand((batch, 1, height, width), device=device, generator=gen, dtype=torch.float32) * 8.0 + 0.2
    invalid = torch.rand((batch, 1, height, width), device=device, generator=gen) < 0.03
    depth = depth.masked_fill(invalid, 0.0)

    w2c = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0).repeat(batch, 1, 1)
    for b in range(batch):
        angle = 0.03 * float(b + 1)
        ca = torch.cos(torch.tensor(angle, device=device))
        sa = torch.sin(torch.tensor(angle, device=device))
        w2c[b, 0, 0] = ca
        w2c[b, 0, 2] = sa
        w2c[b, 2, 0] = -sa
        w2c[b, 2, 2] = ca
        w2c[b, :3, 3] = torch.tensor([0.1 * b, -0.03 * b, 0.2 * b], device=device)

    K = torch.zeros((batch, 3, 3), device=device, dtype=torch.float32)
    K[:, 0, 0] = 0.85 * width
    K[:, 1, 1] = 0.85 * height
    K[:, 0, 2] = 0.5 * width
    K[:, 1, 2] = 0.5 * height
    K[:, 2, 2] = 1.0
    return depth, w2c, K


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and benchmark Sparse3DCache point-building kernels.")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--downsample", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--skip_tilelang", action="store_true", help="Skip the optional TileLang backend.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")

    device = "cuda"
    depth, w2c, K = _make_inputs(args.batch, args.height, args.width, device)

    ref = _reference_depth_to_world(depth, w2c, K, args.downsample)
    tri = depth_to_world_points_triton(depth, w2c, K, args.downsample)
    torch.cuda.synchronize()
    triton_max_abs = float((ref - tri).abs().max().item())
    triton_mean_abs = float((ref - tri).abs().mean().item())

    tile = None
    tilelang_error = None
    if not args.skip_tilelang and depth_to_world_points_tilelang is not None:
        try:
            tile = depth_to_world_points_tilelang(depth, w2c, K, args.downsample)
            torch.cuda.synchronize()
        except Exception as exc:
            tilelang_error = exc
    elif _TILELANG_IMPORT_ERROR is not None:
        tilelang_error = _TILELANG_IMPORT_ERROR

    tilelang_max_abs = None
    tilelang_mean_abs = None
    if tile is not None:
        tilelang_max_abs = float((ref - tile).abs().max().item())
        tilelang_mean_abs = float((ref - tile).abs().mean().item())

    torch_ms = _time_cuda(lambda: _reference_depth_to_world(depth, w2c, K, args.downsample), args.iterations)
    triton_ms = _time_cuda(lambda: depth_to_world_points_triton(depth, w2c, K, args.downsample), args.iterations)
    triton_speedup = torch_ms / max(triton_ms, 1.0e-9)

    tilelang_ms = None
    tilelang_speedup = None
    if tile is not None:
        tilelang_ms = _time_cuda(lambda: depth_to_world_points_tilelang(depth, w2c, K, args.downsample), args.iterations)
        tilelang_speedup = torch_ms / max(tilelang_ms, 1.0e-9)

    print(f"shape: B={args.batch}, H={args.height}, W={args.width}, downsample={args.downsample}")
    print(f"triton_max_abs_diff: {triton_max_abs:.6g}")
    print(f"triton_mean_abs_diff: {triton_mean_abs:.6g}")
    if tilelang_max_abs is not None and tilelang_mean_abs is not None:
        print(f"tilelang_max_abs_diff: {tilelang_max_abs:.6g}")
        print(f"tilelang_mean_abs_diff: {tilelang_mean_abs:.6g}")
    elif tilelang_error is not None:
        print(f"tilelang_error: {type(tilelang_error).__name__}: {tilelang_error}")
    print(f"torch_ms: {torch_ms:.4f}")
    print(f"triton_ms: {triton_ms:.4f}")
    print(f"triton_speedup: {triton_speedup:.2f}x")
    if tilelang_ms is not None and tilelang_speedup is not None:
        print(f"tilelang_ms: {tilelang_ms:.4f}")
        print(f"tilelang_speedup: {tilelang_speedup:.2f}x")


if __name__ == "__main__":
    main()
