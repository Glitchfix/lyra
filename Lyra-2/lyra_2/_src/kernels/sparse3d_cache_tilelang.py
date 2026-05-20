# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TileLang kernels for sparse 3D cache construction."""

from functools import lru_cache

import torch

try:
    import tilelang
    import tilelang.language as T

    TILELANG_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only on systems without TileLang.
    tilelang = None
    T = None
    TILELANG_AVAILABLE = False


def _dtype_to_tilelang(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float32:
        return "float32"
    raise TypeError(f"TileLang Sparse3D cache kernel does not support dtype={dtype}.")


def can_use_tilelang_sparse3d(
    depth_B_1_H_W: torch.Tensor,
    w2c_B_4_4: torch.Tensor,
    K_B_3_3: torch.Tensor,
    downsample: int,
) -> bool:
    """Return whether the fused TileLang cache kernel supports these inputs."""
    if not TILELANG_AVAILABLE or not torch.cuda.is_available():
        return False
    if not depth_B_1_H_W.is_cuda or not w2c_B_4_4.is_cuda or not K_B_3_3.is_cuda:
        return False
    if depth_B_1_H_W.dim() != 4 or depth_B_1_H_W.shape[1] != 1:
        return False
    if w2c_B_4_4.dim() != 3 or tuple(w2c_B_4_4.shape[-2:]) != (4, 4):
        return False
    if K_B_3_3.dim() != 3 or tuple(K_B_3_3.shape[-2:]) != (3, 3):
        return False
    if depth_B_1_H_W.shape[0] != w2c_B_4_4.shape[0] or depth_B_1_H_W.shape[0] != K_B_3_3.shape[0]:
        return False
    if int(downsample) <= 0:
        return False
    if depth_B_1_H_W.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if w2c_B_4_4.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if K_B_3_3.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    return True


if TILELANG_AVAILABLE:

    @tilelang.jit(target="cuda")
    def _depth_to_world_tilelang(
        B: int,
        H: int,
        W: int,
        HDS: int,
        WDS: int,
        DS: int,
        block_size: int = 256,
        depth_dtype: str = "float32",
    ):
        total = B * HDS * WDS

        @T.prim_func
        def kernel(
            depth: T.Tensor((B, 1, H, W), depth_dtype),
            c2w: T.Tensor((B, 4, 4), "float32"),
            K: T.Tensor((B, 3, 3), "float32"),
            out: T.Tensor((B, HDS, WDS, 3), "float32"),
        ):
            with T.Kernel(T.ceildiv(total, block_size), threads=block_size) as bx:
                for i in T.Parallel(block_size):
                    off = bx * block_size + i
                    if off < total:
                        pix = off % (HDS * WDS)
                        b = off // (HDS * WDS)
                        y_ds = pix // WDS
                        x_ds = pix - y_ds * WDS
                        y = y_ds * DS
                        x = x_ds * DS

                        z = T.cast(depth[b, 0, y, x], "float32")
                        if z > 0.0:
                            fx = K[b, 0, 0]
                            fy = K[b, 1, 1]
                            cx = K[b, 0, 2]
                            cy = K[b, 1, 2]

                            x_cam = ((T.cast(x, "float32") - cx) / T.max(T.abs(fx), 0.000001)) * z
                            y_cam = ((T.cast(y, "float32") - cy) / T.max(T.abs(fy), 0.000001)) * z

                            out[b, y_ds, x_ds, 0] = (
                                c2w[b, 0, 0] * x_cam + c2w[b, 0, 1] * y_cam + c2w[b, 0, 2] * z + c2w[b, 0, 3]
                            )
                            out[b, y_ds, x_ds, 1] = (
                                c2w[b, 1, 0] * x_cam + c2w[b, 1, 1] * y_cam + c2w[b, 1, 2] * z + c2w[b, 1, 3]
                            )
                            out[b, y_ds, x_ds, 2] = (
                                c2w[b, 2, 0] * x_cam + c2w[b, 2, 1] * y_cam + c2w[b, 2, 2] * z + c2w[b, 2, 3]
                            )
                        else:
                            out[b, y_ds, x_ds, 0] = 0.0
                            out[b, y_ds, x_ds, 1] = 0.0
                            out[b, y_ds, x_ds, 2] = 0.0

        return kernel


@lru_cache(maxsize=32)
def _compiled_depth_to_world_kernel(
    B: int,
    H: int,
    W: int,
    HDS: int,
    WDS: int,
    DS: int,
    block_size: int,
    depth_dtype: str,
):
    if not TILELANG_AVAILABLE:
        raise RuntimeError("TileLang is not available.")
    return _depth_to_world_tilelang(B, H, W, HDS, WDS, DS, block_size, depth_dtype)


def depth_to_world_points_tilelang(
    depth_B_1_H_W: torch.Tensor,
    w2c_B_4_4: torch.Tensor,
    K_B_3_3: torch.Tensor,
    downsample: int,
    *,
    block_size: int = 256,
) -> torch.Tensor:
    """Build [B, ceil(H/ds), ceil(W/ds), 3] world points with a fused TileLang kernel."""
    if not can_use_tilelang_sparse3d(depth_B_1_H_W, w2c_B_4_4, K_B_3_3, downsample):
        raise RuntimeError("TileLang Sparse3D cache kernel does not support these inputs.")

    B, _C, H, W = depth_B_1_H_W.shape
    ds = int(downsample)
    hds = (int(H) + ds - 1) // ds
    wds = (int(W) + ds - 1) // ds

    depth = depth_B_1_H_W.contiguous()
    c2w = torch.linalg.inv(w2c_B_4_4.to(torch.float32)).contiguous()
    K = K_B_3_3.to(torch.float32).contiguous()
    out = torch.empty((int(B), hds, wds, 3), device=depth.device, dtype=torch.float32)

    kernel = _compiled_depth_to_world_kernel(
        int(B),
        int(H),
        int(W),
        hds,
        wds,
        ds,
        int(block_size),
        _dtype_to_tilelang(depth.dtype),
    )
    kernel(depth, c2w, K, out)
    return out
