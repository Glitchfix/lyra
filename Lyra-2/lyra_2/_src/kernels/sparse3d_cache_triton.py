# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Triton kernels for sparse 3D cache construction.

The cache path repeatedly converts a downsampled depth image into dense world
points. The PyTorch reference path performs the same work through several tensor
ops: slicing, intrinsic scaling, mask construction, unprojection, and c2w
matmul. This module keeps the public contract identical while fusing the
per-pixel work into one coalesced kernel.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only on systems without Triton.
    triton = None
    tl = None
    TRITON_AVAILABLE = False


def can_use_triton_sparse3d(
    depth_B_1_H_W: torch.Tensor,
    w2c_B_4_4: torch.Tensor,
    K_B_3_3: torch.Tensor,
    downsample: int,
) -> bool:
    """Return whether the fused Triton cache kernel supports these inputs."""
    if not TRITON_AVAILABLE or not torch.cuda.is_available():
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


if TRITON_AVAILABLE:

    @triton.jit
    def _depth_to_world_kernel(
        depth,
        c2w,
        K,
        out,
        total: tl.constexpr,
        H: tl.constexpr,
        W: tl.constexpr,
        HDS: tl.constexpr,
        WDS: tl.constexpr,
        DS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < total

        pix = offs % (HDS * WDS)
        b = offs // (HDS * WDS)
        y_ds = pix // WDS
        x_ds = pix - y_ds * WDS
        y = y_ds * DS
        x = x_ds * DS

        depth_idx = ((b * H + y) * W + x)
        z = tl.load(depth + depth_idx, mask=mask, other=0.0).to(tl.float32)
        valid = mask & (z > 0.0)

        k_base = b * 9
        fx = tl.load(K + k_base + 0, mask=mask, other=1.0).to(tl.float32)
        fy = tl.load(K + k_base + 4, mask=mask, other=1.0).to(tl.float32)
        cx = tl.load(K + k_base + 2, mask=mask, other=0.0).to(tl.float32)
        cy = tl.load(K + k_base + 5, mask=mask, other=0.0).to(tl.float32)

        x_cam = ((x.to(tl.float32) - cx) / tl.maximum(tl.abs(fx), 1.0e-6)) * z
        y_cam = ((y.to(tl.float32) - cy) / tl.maximum(tl.abs(fy), 1.0e-6)) * z

        m_base = b * 16
        m00 = tl.load(c2w + m_base + 0, mask=mask, other=0.0).to(tl.float32)
        m01 = tl.load(c2w + m_base + 1, mask=mask, other=0.0).to(tl.float32)
        m02 = tl.load(c2w + m_base + 2, mask=mask, other=0.0).to(tl.float32)
        m03 = tl.load(c2w + m_base + 3, mask=mask, other=0.0).to(tl.float32)
        m10 = tl.load(c2w + m_base + 4, mask=mask, other=0.0).to(tl.float32)
        m11 = tl.load(c2w + m_base + 5, mask=mask, other=0.0).to(tl.float32)
        m12 = tl.load(c2w + m_base + 6, mask=mask, other=0.0).to(tl.float32)
        m13 = tl.load(c2w + m_base + 7, mask=mask, other=0.0).to(tl.float32)
        m20 = tl.load(c2w + m_base + 8, mask=mask, other=0.0).to(tl.float32)
        m21 = tl.load(c2w + m_base + 9, mask=mask, other=0.0).to(tl.float32)
        m22 = tl.load(c2w + m_base + 10, mask=mask, other=0.0).to(tl.float32)
        m23 = tl.load(c2w + m_base + 11, mask=mask, other=0.0).to(tl.float32)

        wx = m00 * x_cam + m01 * y_cam + m02 * z + m03
        wy = m10 * x_cam + m11 * y_cam + m12 * z + m13
        wz = m20 * x_cam + m21 * y_cam + m22 * z + m23

        wx = tl.where(valid, wx, 0.0)
        wy = tl.where(valid, wy, 0.0)
        wz = tl.where(valid, wz, 0.0)

        out_base = offs * 3
        tl.store(out + out_base + 0, wx, mask=mask)
        tl.store(out + out_base + 1, wy, mask=mask)
        tl.store(out + out_base + 2, wz, mask=mask)


def depth_to_world_points_triton(
    depth_B_1_H_W: torch.Tensor,
    w2c_B_4_4: torch.Tensor,
    K_B_3_3: torch.Tensor,
    downsample: int,
    *,
    block_size: int = 256,
) -> torch.Tensor:
    """Build [B, ceil(H/ds), ceil(W/ds), 3] world points with a fused Triton kernel."""
    if not can_use_triton_sparse3d(depth_B_1_H_W, w2c_B_4_4, K_B_3_3, downsample):
        raise RuntimeError("Triton Sparse3D cache kernel does not support these inputs.")

    B, _C, H, W = depth_B_1_H_W.shape
    ds = int(downsample)
    hds = (int(H) + ds - 1) // ds
    wds = (int(W) + ds - 1) // ds
    total = int(B) * hds * wds

    depth = depth_B_1_H_W.contiguous()
    c2w = torch.linalg.inv(w2c_B_4_4.to(torch.float32)).contiguous()
    K = K_B_3_3.to(torch.float32).contiguous()
    out = torch.empty((int(B), hds, wds, 3), device=depth.device, dtype=torch.float32)

    grid = (triton.cdiv(total, int(block_size)),)
    _depth_to_world_kernel[grid](
        depth,
        c2w,
        K,
        out,
        total,
        int(H),
        int(W),
        hds,
        wds,
        ds,
        BLOCK=int(block_size),
    )
    return out
