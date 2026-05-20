# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single-image → video generation along a user-supplied camera trajectory.

Pipeline:
1. Read a single input image.
2. Load per-chunk text captions from a captions.json file (or single --prompt).
3. Load camera trajectory (w2c + intrinsics) from an .npz file,
   take the first ``num_frames`` poses.
4. Produce a video using FramePack AR spatial generation with per-chunk T5 embeddings.
5. Save the output video.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import select
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from lyra_2._ext.imaginaire.utils import log, misc
from lyra_2._ext.imaginaire.visualize.video import save_img_or_video
from lyra_2._src.inference.lyra2_ar_inference import (
    _offload_diffusion_to_cpu,
    _offload_module_to_cpu,
    _restore_diffusion_to_gpu,
    _restore_module_to_device,
    save_output,
    safe_to,
    run_lyra2_sample,
)
from lyra_2._src.inference.lyra2_zoomgs_inference import (
    _da3_infer_depth_intrinsics_single,
    _build_image_list,
)
from lyra_2._src.utils.model_loader import load_model_from_checkpoint

torch.enable_grad(False)
torch.backends.cudnn.enabled = False


def _get_t5_embedding_memory_safe(caption, args, desired_device, desired_dtype, model):
    from lyra_2._src.inference.get_t5_emb import get_umt5_embedding, get_umt5_embedding_offloaded

    if args.offload_when_prompt:
        _offload_diffusion_to_cpu(model, True)
    try:
        if args.offload_when_prompt:
            emb = get_umt5_embedding_offloaded(caption, device=desired_device)
        else:
            emb = get_umt5_embedding(caption, device=desired_device)
    finally:
        if args.offload_when_prompt:
            _restore_diffusion_to_gpu(model, True)
    return emb.to(dtype=desired_dtype)


# ---------------------------------------------------------------------------
# Trajectory loading
# ---------------------------------------------------------------------------

def load_trajectory(
    path: str,
    num_frames: int,
    target_hw: tuple[int, int] | None = None,
    pose_scale: float = 1.0,
    extend_mode: str = "error",
):
    """Load camera trajectory from an .npz file.

    Expected keys:
        w2c        – (N, 4, 4) world-to-camera matrices  (float32/64)
        intrinsics – (N, 3, 3) camera intrinsic matrices  (float32/64)
        image_height, image_width – original resolution the intrinsics refer to

    If *target_hw* is provided and differs from the stored resolution,
    intrinsics are rescaled accordingly.

    Returns the first *num_frames* entries as torch tensors.
    """
    data = np.load(path)
    w2c_np = data["w2c"].astype(np.float32)
    intr_np = data["intrinsics"].astype(np.float32)
    available = int(w2c_np.shape[0])
    if num_frames <= available:
        indices = np.arange(num_frames)
    elif extend_mode == "loop":
        indices = np.arange(num_frames) % available
    elif extend_mode == "repeat_last":
        indices = np.concatenate(
            [np.arange(available), np.full(num_frames - available, available - 1, dtype=np.int64)]
        )
    else:
        raise ValueError(
            f"Trajectory has {available} frames but {num_frames} were requested. "
            "Use --trajectory_extend loop or repeat_last for continuous generation."
        )

    w2c = torch.from_numpy(w2c_np[indices])
    intrinsics = torch.from_numpy(intr_np[indices])

    if pose_scale != 1.0:
        w2c[:, :3, 3] *= pose_scale

    if target_hw is not None and "image_height" in data and "image_width" in data:
        orig_h, orig_w = int(data["image_height"]), int(data["image_width"])
        tgt_h, tgt_w = target_hw
        if (orig_h, orig_w) != (tgt_h, tgt_w):
            sx = tgt_w / orig_w
            sy = tgt_h / orig_h
            intrinsics[:, 0, 0] *= sx
            intrinsics[:, 0, 2] *= sx
            intrinsics[:, 1, 1] *= sy
            intrinsics[:, 1, 2] *= sy

    return w2c, intrinsics


def _depth_to_sparse_points(
    depth_hw: torch.Tensor,
    K_33: torch.Tensor,
    mask_hw: torch.Tensor,
    *,
    stride: int,
    max_points: int,
    depth_percentile: float,
) -> np.ndarray:
    depth = depth_hw.detach().to(dtype=torch.float32, device="cpu").numpy()
    mask = mask_hw.detach().to(dtype=torch.float32, device="cpu").numpy() > 0.5
    valid = np.isfinite(depth) & (depth > 1e-4) & (depth < 1e4) & mask
    if not np.any(valid):
        raise ValueError("Cannot build WoS trajectory: no valid depth pixels.")

    cap = np.percentile(depth[valid], float(depth_percentile))
    valid &= depth <= cap

    H, W = depth.shape
    step = max(1, int(stride))
    yy, xx = np.mgrid[0:H:step, 0:W:step]
    yy = yy.reshape(-1)
    xx = xx.reshape(-1)
    keep = valid[yy, xx]
    yy = yy[keep]
    xx = xx[keep]
    if yy.size == 0:
        yy, xx = np.nonzero(valid)

    if int(max_points) > 0 and yy.size > int(max_points):
        sel = np.linspace(0, yy.size - 1, int(max_points), dtype=np.int64)
        yy = yy[sel]
        xx = xx[sel]

    K = K_33.detach().to(dtype=torch.float32, device="cpu").numpy()
    z = depth[yy, xx].astype(np.float32)
    x = ((xx.astype(np.float32) - K[0, 2]) / max(float(K[0, 0]), 1e-6)) * z
    y = ((yy.astype(np.float32) - K[1, 2]) / max(float(K[1, 1]), 1e-6)) * z
    return np.stack([x, y, z], axis=1).astype(np.float32)


def _center_depth_from_depth(depth_hw: torch.Tensor, mask_hw: torch.Tensor) -> float:
    depth = depth_hw.detach().to(dtype=torch.float32, device="cpu").numpy()
    mask = mask_hw.detach().to(dtype=torch.float32, device="cpu").numpy() > 0.5
    H, W = depth.shape
    y0, y1 = int(0.30 * H), int(0.70 * H)
    x0, x1 = int(0.30 * W), int(0.70 * W)
    center = depth[y0:y1, x0:x1]
    center_mask = mask[y0:y1, x0:x1]
    valid = center[np.isfinite(center) & (center > 1e-4) & (center < 1e4) & center_mask]
    if valid.size < 16:
        valid = depth[np.isfinite(depth) & (depth > 1e-4) & (depth < 1e4) & mask]
    if valid.size == 0:
        raise ValueError("Cannot build WoS trajectory: no valid center depth.")
    return float(np.median(valid))


def _nearest_point_distance(pos: np.ndarray, points: np.ndarray) -> tuple[float, np.ndarray]:
    diff = points - pos[None, :]
    d2 = np.einsum("ij,ij->i", diff, diff)
    idx = int(np.argmin(d2))
    return float(math.sqrt(max(float(d2[idx]), 0.0))), points[idx]


def _wos_adjust_path(
    desired: np.ndarray,
    points: np.ndarray,
    *,
    clearance: float,
    step_fraction: float,
    max_steps_per_frame: int = 64,
) -> np.ndarray:
    planned = np.empty_like(desired, dtype=np.float32)
    planned[0] = desired[0]
    step_fraction = float(np.clip(step_fraction, 0.1, 1.0))
    clearance = max(float(clearance), 1e-5)

    for i in range(1, desired.shape[0] - 1):
        cur = planned[i - 1].astype(np.float32).copy()
        goal = desired[i].astype(np.float32)
        for _ in range(max_steps_per_frame):
            to_goal = goal - cur
            dist_to_goal = float(np.linalg.norm(to_goal))
            if dist_to_goal < 1e-5:
                cur = goal
                break
            nn_dist, nearest = _nearest_point_distance(cur, points)
            free_radius = nn_dist - clearance
            if free_radius <= 0.0:
                away = cur - nearest
                away_norm = float(np.linalg.norm(away))
                if away_norm < 1e-6:
                    away = -to_goal
                    away_norm = float(np.linalg.norm(away))
                cur = cur + away / max(away_norm, 1e-6) * (clearance - nn_dist + 1e-4)
                continue
            step = min(dist_to_goal, max(free_radius * step_fraction, clearance * 0.05))
            cur = cur + to_goal / dist_to_goal * step

        nn_dist, nearest = _nearest_point_distance(cur, points)
        if nn_dist < clearance:
            away = cur - nearest
            away_norm = float(np.linalg.norm(away))
            if away_norm > 1e-6:
                cur = cur + away / away_norm * (clearance - nn_dist + 1e-4)
        planned[i] = cur

    planned[-1] = desired[-1]
    for _ in range(2):
        smoothed = planned.copy()
        smoothed[1:-1] = 0.25 * planned[:-2] + 0.5 * planned[1:-1] + 0.25 * planned[2:]
        planned = smoothed
        planned[0] = desired[0]
        planned[-1] = desired[-1]
    return planned


def _look_at_w2c_np(camera_pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = target - camera_pos
    forward = forward / max(float(np.linalg.norm(forward)), 1e-8)
    up_hint = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(up_hint, forward))) > 0.98:
        up_hint = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    right = np.cross(up_hint, forward)
    right = right / max(float(np.linalg.norm(right)), 1e-8)
    up = np.cross(forward, right)
    up = up / max(float(np.linalg.norm(up)), 1e-8)

    w2c = np.eye(4, dtype=np.float32)
    w2c[0, :3] = right
    w2c[1, :3] = up
    w2c[2, :3] = forward
    w2c[:3, 3] = -w2c[:3, :3] @ camera_pos.astype(np.float32)
    return w2c


def _short_angle_delta(target: float, current: float) -> float:
    return (float(target) - float(current) + math.pi) % (2.0 * math.pi) - math.pi


def _json_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _json_vec3(value) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    out = [_json_float(item, float("nan")) for item in value]
    if not np.isfinite(np.asarray(out, dtype=np.float32)).all():
        return None
    return out


def _command_history_record(chunk_index: int, cmd: dict) -> dict:
    record: dict[str, object] = {"chunk": float(chunk_index)}
    for key, value in cmd.items():
        if key == "trail" and isinstance(value, list):
            record["trail_samples"] = float(len(value))
        elif isinstance(value, (list, tuple, np.ndarray)):
            record[key] = [_json_float(item) for item in value]
        elif isinstance(value, (int, float, np.integer, np.floating)):
            record[key] = float(value)
        elif value is not None:
            record[key] = str(value)
    return record


def _free_roam_trail_from_data(data: dict) -> list[dict]:
    raw = data.get("trail")
    if not isinstance(raw, list):
        return []
    trail = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        pos = _json_vec3(item.get("position"))
        if pos is None:
            continue
        trail.append(
            {
                "sequence": _json_float(item.get("sequence"), -1.0),
                "position": pos,
                "yaw_degrees": _json_float(item.get("yaw_degrees"), 0.0),
                "pitch_degrees": _json_float(item.get("pitch_degrees"), 0.0),
            }
        )
    trail.sort(key=lambda item: float(item["sequence"]))
    return trail


def _interp_path(values: np.ndarray, total: int) -> np.ndarray:
    if values.shape[0] <= 1:
        return np.repeat(values[:1], int(total), axis=0)
    src = np.linspace(0.0, 1.0, values.shape[0], dtype=np.float32)
    dst = np.linspace(0.0, 1.0, int(total), dtype=np.float32)
    out = np.empty((int(total), values.shape[1]), dtype=np.float32)
    for dim in range(values.shape[1]):
        out[:, dim] = np.interp(dst, src, values[:, dim]).astype(np.float32)
    return out


def build_wos_360_trajectory(
    depth_hw: torch.Tensor,
    K_33: torch.Tensor,
    mask_hw: torch.Tensor,
    *,
    num_frames: int,
    loops: float,
    radius_scale: float,
    vertical_amp: float,
    clearance_ratio: float,
    step_fraction: float,
    depth_stride: int,
    max_points: int,
    depth_percentile: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    if num_frames < 2:
        raise ValueError("WoS 360 trajectory requires at least two frames.")

    center_depth = _center_depth_from_depth(depth_hw, mask_hw)
    points = _depth_to_sparse_points(
        depth_hw,
        K_33,
        mask_hw,
        stride=depth_stride,
        max_points=max_points,
        depth_percentile=depth_percentile,
    )

    target = np.array([0.0, 0.0, center_depth], dtype=np.float32)
    theta = np.linspace(0.0, 2.0 * math.pi * float(loops), int(num_frames), endpoint=True, dtype=np.float32)
    frac = np.linspace(0.0, 1.0, int(num_frames), endpoint=True, dtype=np.float32)

    radius_base = center_depth
    radius = radius_base * (1.0 + (float(radius_scale) - 1.0) * (np.sin(math.pi * frac) ** 2))
    desired = np.zeros((int(num_frames), 3), dtype=np.float32)
    desired[:, 0] = radius * np.sin(theta)
    desired[:, 1] = float(vertical_amp) * center_depth * np.sin(2.0 * theta)
    desired[:, 2] = center_depth - radius * np.cos(theta)
    desired[0] = 0.0
    desired[-1] = 0.0

    clearance = max(float(clearance_ratio) * center_depth, 1e-4)
    planned = _wos_adjust_path(
        desired,
        points,
        clearance=clearance,
        step_fraction=step_fraction,
    )
    planned[0] = 0.0
    planned[-1] = 0.0

    w2c_np = np.stack([_look_at_w2c_np(p, target) for p in planned], axis=0)
    w2c_np[0] = np.eye(4, dtype=np.float32)
    w2c_np[-1] = np.eye(4, dtype=np.float32)
    Ks = K_33.detach().to(dtype=torch.float32, device="cpu").unsqueeze(0).repeat(int(num_frames), 1, 1)
    stats = {
        "center_depth": float(center_depth),
        "clearance": float(clearance),
        "num_points": float(points.shape[0]),
        "loops": float(loops),
    }
    return torch.from_numpy(w2c_np.astype(np.float32)), Ks, stats


def build_wos_lookaround_trajectory(
    depth_hw: torch.Tensor,
    K_33: torch.Tensor,
    mask_hw: torch.Tensor,
    *,
    num_frames: int,
    yaw_degrees: float,
    pitch_degrees: float,
    pitch_cycles: float,
    translation_ratio: float,
    clearance_ratio: float,
    step_fraction: float,
    depth_stride: int,
    max_points: int,
    depth_percentile: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    if num_frames < 2:
        raise ValueError("WoS look-around trajectory requires at least two frames.")

    center_depth = _center_depth_from_depth(depth_hw, mask_hw)
    points = _depth_to_sparse_points(
        depth_hw,
        K_33,
        mask_hw,
        stride=depth_stride,
        max_points=max_points,
        depth_percentile=depth_percentile,
    )

    frac = np.linspace(0.0, 1.0, int(num_frames), endpoint=True, dtype=np.float32)
    # Smooth start/end reduces the early-frame snap that tends to destabilize generation.
    eased = frac * frac * (3.0 - 2.0 * frac)
    yaw = np.deg2rad(float(yaw_degrees)) * eased
    pitch_amp = np.deg2rad(float(pitch_degrees))
    pitch = pitch_amp * np.sin(2.0 * math.pi * float(pitch_cycles) * eased)

    drift = max(float(translation_ratio), 0.0) * center_depth
    desired = np.zeros((int(num_frames), 3), dtype=np.float32)
    desired[:, 0] = drift * np.sin(2.0 * math.pi * eased)
    desired[:, 1] = 0.5 * drift * np.sin(4.0 * math.pi * eased)
    desired[:, 2] = 0.25 * drift * (1.0 - np.cos(2.0 * math.pi * eased))
    desired[0] = 0.0
    desired[-1] = 0.0

    clearance = max(float(clearance_ratio) * center_depth, 1e-4)
    planned = _wos_adjust_path(
        desired,
        points,
        clearance=clearance,
        step_fraction=step_fraction,
    )
    planned[0] = 0.0
    planned[-1] = 0.0

    w2c_list = []
    for p, y, pt in zip(planned, yaw, pitch):
        forward = np.array(
            [
                math.sin(float(y)) * math.cos(float(pt)),
                math.sin(float(pt)),
                math.cos(float(y)) * math.cos(float(pt)),
            ],
            dtype=np.float32,
        )
        target = p + forward * center_depth
        w2c_list.append(_look_at_w2c_np(p, target))

    w2c_np = np.stack(w2c_list, axis=0).astype(np.float32)
    w2c_np[0] = np.eye(4, dtype=np.float32)
    w2c_np[-1] = np.eye(4, dtype=np.float32)
    Ks = K_33.detach().to(dtype=torch.float32, device="cpu").unsqueeze(0).repeat(int(num_frames), 1, 1)
    drift_norm = np.linalg.norm(planned, axis=1)
    stats = {
        "center_depth": float(center_depth),
        "clearance": float(clearance),
        "num_points": float(points.shape[0]),
        "yaw_degrees": float(yaw_degrees),
        "pitch_degrees": float(pitch_degrees),
        "max_drift": float(drift_norm.max() if drift_norm.size else 0.0),
    }
    return torch.from_numpy(w2c_np), Ks, stats


class WosWalkTrajectoryStream:
    """Chunk-wise WoS trajectory planner for long walking-style generation.

    The planner emits only the next AR chunk of poses. It keeps a small sparse
    obstacle map and can refresh it from Lyra's spatial cache after each chunk.
    """

    def __init__(
        self,
        depth_hw: torch.Tensor,
        K_33: torch.Tensor,
        mask_hw: torch.Tensor,
        *,
        seed: int,
        speed_ratio: float,
        turn_degrees_per_chunk: float,
        random_turn_degrees: float,
        strafe_ratio: float,
        bob_ratio: float,
        pitch_degrees: float,
        vertical_ratio: float,
        free_roam_max_chunk_ratio: float,
        clearance_ratio: float,
        step_fraction: float,
        depth_stride: int,
        max_points: int,
        depth_percentile: float,
        control_mode: str,
        control_path: str | None,
        map_update_max_points: int,
    ) -> None:
        self.K = K_33.detach().to(dtype=torch.float32, device="cpu").numpy().astype(np.float32)
        self.K_torch = K_33.detach().to(dtype=torch.float32, device="cpu")
        self.center_depth = _center_depth_from_depth(depth_hw, mask_hw)
        self.seed_points = _depth_to_sparse_points(
            depth_hw,
            K_33,
            mask_hw,
            stride=depth_stride,
            max_points=max_points,
            depth_percentile=depth_percentile,
        )
        self.points = self.seed_points.copy()
        self.clearance = max(float(clearance_ratio) * self.center_depth, 1e-4)
        self.step_fraction = float(step_fraction)
        self.speed = max(float(speed_ratio), 0.0) * self.center_depth
        self.vertical_speed = max(float(vertical_ratio), 0.0) * self.center_depth
        self.free_roam_max_delta = max(float(free_roam_max_chunk_ratio), 0.0) * self.center_depth
        self.max_turn = math.radians(abs(float(turn_degrees_per_chunk)))
        self.random_turn = math.radians(abs(float(random_turn_degrees)))
        self.strafe = max(float(strafe_ratio), 0.0) * self.center_depth
        self.bob = max(float(bob_ratio), 0.0) * self.center_depth
        self.pitch_amp = math.radians(abs(float(pitch_degrees)))
        self.control_mode = str(control_mode)
        self.control_path = control_path
        self.map_update_max_points = int(map_update_max_points)
        self.rng = np.random.default_rng(int(seed))

        self.position = np.zeros(3, dtype=np.float32)
        self.yaw = 0.0
        self.pitch = 0.0
        self.turn_velocity = 0.0
        self.frame_index = 0
        self.command = {"forward": 1.0, "strafe": 0.0, "vertical": 0.0, "turn": 0.0, "pitch": 0.0}
        if self.control_mode == "free_roam":
            self.command["forward"] = 0.0
        self.free_roam_last_sequence = -1.0
        self.generated_w2c: list[np.ndarray] = [np.eye(4, dtype=np.float32)]
        self.generated_K: list[np.ndarray] = [self.K.copy()]
        self.generated_positions: list[np.ndarray] = [self.position.copy()]
        self.command_history: list[dict[str, object]] = []

    def _command_from_file(self) -> dict[str, float] | None:
        if not self.control_path or not os.path.isfile(self.control_path):
            return None
        try:
            with open(self.control_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
        if isinstance(data, list):
            data = data[-1] if data else {}
        if not isinstance(data, dict):
            return None
        cmd = {
            "forward": _json_float(data.get("forward", self.command["forward"]), self.command["forward"]),
            "strafe": _json_float(data.get("strafe", self.command["strafe"]), self.command["strafe"]),
            "vertical": _json_float(
                data.get("vertical", self.command.get("vertical", 0.0)),
                self.command.get("vertical", 0.0),
            ),
            "turn": _json_float(data.get("turn", data.get("turn_degrees", self.command["turn"])), self.command["turn"]),
            "pitch": _json_float(data.get("pitch", data.get("pitch_degrees", self.command["pitch"])), self.command["pitch"]),
        }
        target_position = _json_vec3(data.get("target_position", data.get("position")))
        if target_position is not None:
            cmd["target_position"] = target_position
        if "yaw_degrees" in data or "yaw" in data:
            cmd["target_yaw"] = _json_float(data.get("yaw_degrees", data.get("yaw")), math.degrees(self.yaw))
        if "pitch_degrees" in data or "absolute_pitch" in data:
            cmd["target_pitch"] = _json_float(
                data.get("pitch_degrees", data.get("absolute_pitch")),
                math.degrees(self.pitch),
            )
        if "sequence" in data:
            cmd["sequence"] = _json_float(data.get("sequence"), 0.0)
        trail = _free_roam_trail_from_data(data)
        if trail:
            cmd["trail"] = trail
        if "mode" in data:
            cmd["mode"] = str(data.get("mode"))
        return cmd

    def _command_from_keyboard(self) -> dict[str, float] | None:
        if not sys.stdin or not hasattr(sys.stdin, "fileno"):
            return None
        try:
            ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        except (OSError, ValueError):
            return None
        if not ready:
            return None
        line = sys.stdin.readline().strip().lower()
        if not line:
            return None
        if line in {"random", "auto"}:
            self.control_mode = "random"
            return None
        cmd = {"forward": 0.0, "strafe": 0.0, "vertical": 0.0, "turn": 0.0, "pitch": 0.0}
        if "w" in line:
            cmd["forward"] += 1.0
        if "s" in line:
            cmd["forward"] -= 0.5
        if "a" in line:
            cmd["strafe"] -= 1.0
        if "d" in line:
            cmd["strafe"] += 1.0
        if "z" in line:
            cmd["vertical"] -= 1.0
        if "c" in line:
            cmd["vertical"] += 1.0
        if "q" in line:
            cmd["turn"] -= math.degrees(self.max_turn)
        if "e" in line:
            cmd["turn"] += math.degrees(self.max_turn)
        if "r" in line:
            cmd["pitch"] += math.degrees(self.pitch_amp)
        if "f" in line:
            cmd["pitch"] -= math.degrees(self.pitch_amp)
        if "x" in line:
            cmd = {"forward": 0.0, "strafe": 0.0, "vertical": 0.0, "turn": 0.0, "pitch": 0.0}
        return cmd

    def _next_command(self) -> dict[str, float]:
        if self.control_mode in {"file", "free_roam"}:
            cmd = self._command_from_file()
            if cmd is not None:
                self.command = cmd
        elif self.control_mode == "keyboard":
            cmd = self._command_from_keyboard()
            if cmd is not None:
                self.command = cmd
        else:
            noise = float(self.rng.normal(0.0, self.random_turn))
            self.turn_velocity = float(np.clip(0.85 * self.turn_velocity + 0.15 * noise, -self.max_turn, self.max_turn))
            self.command = {
                "forward": 1.0,
                "strafe": float(self.rng.normal(0.0, 0.20)),
                "vertical": 0.0,
                "turn": math.degrees(self.turn_velocity),
                "pitch": float(self.rng.normal(0.0, math.degrees(self.pitch_amp) * 0.25)),
            }
        return dict(self.command)

    def next_chunk(self, num_frames: int, *, chunk_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        cmd = self._next_command()
        self.command_history.append(_command_history_record(chunk_index, cmd))

        total = int(num_frames) + 2
        frac = np.linspace(0.0, 1.0, total, endpoint=True, dtype=np.float32)

        desired = np.empty((total, 3), dtype=np.float32)
        desired[0] = self.position
        trail = cmd.get("trail")
        new_trail = []
        if isinstance(trail, list) and trail:
            for item in trail:
                if not isinstance(item, dict):
                    continue
                seq = _json_float(item.get("sequence"), -1.0)
                pos = _json_vec3(item.get("position"))
                if pos is None or seq <= self.free_roam_last_sequence:
                    continue
                new_trail.append(
                    {
                        "sequence": seq,
                        "position": pos,
                        "yaw_degrees": _json_float(item.get("yaw_degrees"), math.degrees(self.yaw)),
                        "pitch_degrees": _json_float(item.get("pitch_degrees"), math.degrees(self.pitch)),
                    }
                )
            new_trail.sort(key=lambda item: float(item["sequence"]))
            # On startup, avoid replaying minutes of stale controller history.
            if self.free_roam_last_sequence < 0.0 and len(new_trail) > 32:
                new_trail = new_trail[-32:]

        if new_trail:
            self.free_roam_last_sequence = max(float(item["sequence"]) for item in new_trail)
            positions = np.asarray([self.position] + [item["position"] for item in new_trail], dtype=np.float32)
            final_delta = positions[-1] - self.position
            final_delta_norm = float(np.linalg.norm(final_delta))
            max_delta = self.free_roam_max_delta
            if max_delta > 0.0 and final_delta_norm > max_delta:
                positions = self.position[None, :] + (positions - self.position[None, :]) * (max_delta / final_delta_norm)
            desired = _interp_path(positions, total)
            if self.bob > 0.0:
                bob = self.bob * np.sin(2.0 * math.pi * (self.frame_index + np.arange(total)) / 24.0)
                desired[:, 1] += bob.astype(np.float32)

            yaw_values = np.unwrap(np.deg2rad([math.degrees(self.yaw)] + [item["yaw_degrees"] for item in new_trail]))
            pitch_values = np.deg2rad([math.degrees(self.pitch)] + [item["pitch_degrees"] for item in new_trail])
            src = np.linspace(0.0, 1.0, len(yaw_values), dtype=np.float32)
            yaw = np.interp(frac, src, yaw_values).astype(np.float32)
            pitch = np.interp(frac, src, pitch_values).astype(np.float32)
        elif "target_position" in cmd:
            target = np.asarray(cmd["target_position"], dtype=np.float32)
            delta = target - self.position
            delta_norm = float(np.linalg.norm(delta))
            max_delta = self.free_roam_max_delta
            if max_delta <= 0.0:
                max_delta = max(self.speed * float(num_frames), self.clearance)
            if delta_norm > max_delta:
                target = self.position + delta / max(delta_norm, 1e-6) * max_delta
            desired = self.position[None, :] + (target - self.position)[None, :] * frac[:, None]
            if self.bob > 0.0:
                bob = self.bob * np.sin(2.0 * math.pi * (self.frame_index + np.arange(total)) / 24.0)
                desired[:, 1] += bob.astype(np.float32)

            target_yaw = math.radians(_json_float(cmd.get("target_yaw"), math.degrees(self.yaw)))
            target_pitch = math.radians(_json_float(cmd.get("target_pitch"), math.degrees(self.pitch)))
            yaw = self.yaw + _short_angle_delta(target_yaw, self.yaw) * frac
            pitch = self.pitch + (target_pitch - self.pitch) * frac
        else:
            turn_total = math.radians(_json_float(cmd.get("turn"), 0.0))
            pitch_target = math.radians(_json_float(cmd.get("pitch"), math.degrees(self.pitch)))
            yaw = self.yaw + turn_total * frac
            pitch = self.pitch + (pitch_target - self.pitch) * frac

            for i in range(1, total):
                heading = np.array([math.sin(float(yaw[i])), 0.0, math.cos(float(yaw[i]))], dtype=np.float32)
                right = np.array([math.cos(float(yaw[i])), 0.0, -math.sin(float(yaw[i]))], dtype=np.float32)
                up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                step = self.speed * _json_float(cmd.get("forward"), 0.0)
                strafe = self.strafe * _json_float(cmd.get("strafe"), 0.0) * math.sin(2.0 * math.pi * float(frac[i]))
                vertical = self.vertical_speed * _json_float(cmd.get("vertical"), 0.0)
                bob = self.bob * math.sin(2.0 * math.pi * (self.frame_index + i) / 24.0)
                desired[i] = desired[i - 1] + heading * step + right * strafe + up * vertical
                desired[i, 1] += bob

        planned = _wos_adjust_path(
            desired,
            self.points,
            clearance=self.clearance,
            step_fraction=self.step_fraction,
        )

        out_positions = planned[1 : int(num_frames) + 1]
        out_yaw = yaw[1 : int(num_frames) + 1]
        out_pitch = pitch[1 : int(num_frames) + 1]
        w2c = []
        for p, y, pt in zip(out_positions, out_yaw, out_pitch):
            forward = np.array(
                [
                    math.sin(float(y)) * math.cos(float(pt)),
                    math.sin(float(pt)),
                    math.cos(float(y)) * math.cos(float(pt)),
                ],
                dtype=np.float32,
            )
            target = p + forward * self.center_depth
            w2c.append(_look_at_w2c_np(p, target))

        self.position = planned[int(num_frames)].astype(np.float32)
        self.yaw = float(yaw[int(num_frames)])
        self.pitch = float(pitch[int(num_frames)])
        self.frame_index += int(num_frames)
        w2c_np = np.stack(w2c, axis=0).astype(np.float32)
        K_np = np.repeat(self.K[None], int(num_frames), axis=0).astype(np.float32)
        self.generated_w2c.extend(w2c_np)
        self.generated_K.extend(K_np)
        self.generated_positions.extend(out_positions.astype(np.float32))
        return torch.from_numpy(w2c_np), torch.from_numpy(K_np)

    def update_from_pipeline(self, pipeline) -> None:
        cache = getattr(pipeline, "retrieval_cache", None)
        if cache is None or not getattr(cache, "_world_points", None):
            return
        pts_parts = [self.seed_points]
        per_entry_cap = max(256, self.map_update_max_points // max(1, len(cache._world_points)))
        for world_t in cache._world_points:
            pts = world_t.detach().to(torch.float32).cpu().numpy().reshape(-1, 3)
            valid = np.isfinite(pts).all(axis=1) & (np.linalg.norm(pts, axis=1) > 1e-5)
            pts = pts[valid]
            if pts.shape[0] > per_entry_cap:
                sel = np.linspace(0, pts.shape[0] - 1, per_entry_cap, dtype=np.int64)
                pts = pts[sel]
            if pts.size:
                pts_parts.append(pts.astype(np.float32))
        points = np.concatenate(pts_parts, axis=0)
        if points.shape[0] > self.map_update_max_points:
            sel = np.linspace(0, points.shape[0] - 1, self.map_update_max_points, dtype=np.int64)
            points = points[sel]
        self.points = points.astype(np.float32)

    def save_trajectory(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez(
            path,
            w2c=np.stack(self.generated_w2c, axis=0).astype(np.float32),
            intrinsics=np.stack(self.generated_K, axis=0).astype(np.float32),
            positions=np.stack(self.generated_positions, axis=0).astype(np.float32),
            image_height=np.array(int(self.K[1, 2] * 2), dtype=np.int64),
            image_width=np.array(int(self.K[0, 2] * 2), dtype=np.int64),
            center_depth=np.array(self.center_depth, dtype=np.float32),
            clearance=np.array(self.clearance, dtype=np.float32),
            sparse_points=np.array(self.points.shape[0], dtype=np.int64),
            command_history=np.array(json.dumps(self.command_history)),
        )


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-image video generation with a custom camera trajectory"
    )
    # Input
    parser.add_argument("--input_image_path", type=str, required=True,
                        help="Path to a single image or a folder of images")
    parser.add_argument("--trajectory_path", type=str, default=None,
                        help="Path to .npz trajectory file (or a folder of per-image .npz files). "
                             "Expected keys: w2c (N,4,4), intrinsics (N,3,3), "
                             "image_height, image_width.")
    parser.add_argument("--trajectory_preset", type=str, default="file",
                        choices=["file", "wos_360", "wos_lookaround", "wos_walk"],
                        help="Use 'file' to load --trajectory_path, 'wos_360' for a large depth-aware "
                             "orbit, 'wos_lookaround' for slow yaw/pitch scanning, or 'wos_walk' "
                             "for chunk-wise walking traversal.")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--sample_start_idx", type=int, default=0)
    parser.add_argument("--prompt", type=str, default="",
                        help="Optional explicit prompt applied to ALL images (single caption).")
    parser.add_argument("--prompt_dir", type=str, default=None,
                        help="Directory containing per-image .txt caption files (single caption).")
    parser.add_argument("--captions_path", type=str, default=None,
                        help="Path to captions.json (or dir with per-image .json files). "
                             "JSON maps frame-index strings to caption text. "
                             "Each AR chunk uses the caption whose key is <= current frame.")
    parser.add_argument("--prompt_suffix", type=str, default="",
                        help="Text appended to every prompt.")

    # Model and generation
    parser.add_argument("--experiment", type=str, default="lyra_framepack_spatial")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/model")
    parser.add_argument("--output_path", type=str, default="inference/lyra2_custom_traj")
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--num_sampling_step", type=int, default=35)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--num_frames", type=int, default=161,
                        help="Number of frames to generate (taken from the start of the trajectory).")
    parser.add_argument("--continuous_chunks", type=int, default=0,
                        help="Generate this many AR chunks in streaming/ring mode. "
                             "Overrides --num_frames as 1 + chunks * new_video_frames.")
    parser.add_argument("--unbounded_stream", action="store_true",
                        help="Keep generating streaming chunks until interrupted. Use --stream_max_chunks to cap tests.")
    parser.add_argument("--stream_max_chunks", type=int, default=0,
                        help="Safety cap for --unbounded_stream. 0 means no cap.")
    parser.add_argument("--trajectory_extend", type=str, default="error",
                        choices=["error", "loop", "repeat_last"],
                        help="How to extend a short trajectory when continuous generation needs more poses.")
    parser.add_argument("--stream_output_chunks", action="store_true",
                        help="Save each generated AR chunk immediately instead of returning one full video tensor.")
    parser.add_argument("--ring_history_latents", type=int, default=0,
                        help="Recent latent history entries to keep in continuous mode. "
                             "0 uses the model's minimum temporal history.")
    parser.add_argument("--ring_history_frames", type=int, default=0,
                        help="Recent pixel frames to keep in continuous mode. "
                             "0 keeps enough for DA3 plus one generated chunk.")
    parser.add_argument("--ring_cache_entries", type=int, default=32,
                        help="Recent positive frame IDs to keep in Sparse3DCache in continuous mode.")
    parser.add_argument("--ring_anchor_stride", type=int, default=0,
                        help="Keep every Nth frame as a bounded landmark anchor in the ring cache. 0 disables.")
    parser.add_argument("--ring_anchor_entries", type=int, default=12,
                        help="Maximum periodic landmark anchors retained when --ring_anchor_stride is set.")
    parser.add_argument("--stream_export_map_every", type=int, default=0,
                        help="Export sparse stitched PLY map every N generated chunks. 0 disables map export.")
    parser.add_argument("--stream_map_max_points", type=int, default=200000,
                        help="Maximum points written to each sparse stitched map PLY.")
    parser.add_argument("--stream_preview_frames", action="store_true",
                        help="Write per-frame preview images plus manifest.json for the GL free-roam viewer.")
    parser.add_argument("--stream_preview_dir", type=str, default=None,
                        help="Preview frame directory. Defaults to <stream_output_dir>/preview when enabled.")
    parser.add_argument("--cache_kernel", type=str, default="torch", choices=["torch", "triton", "tilelang", "auto"],
                        help="Sparse3DCache point-building backend. 'triton' and 'tilelang' fuse depth downsample, "
                             "unprojection, and camera transform; 'auto' tries TileLang first, then Triton.")
    parser.add_argument("--pose_scale", type=float, default=1.1,
                        help="Scale factor applied to w2c translation vectors.")
    parser.add_argument("--wos_360_loops", type=float, default=1.0,
                        help="Number of horizontal 360 loops for --trajectory_preset wos_360.")
    parser.add_argument("--wos_360_radius_scale", type=float, default=1.0,
                        help="Mid-orbit radius multiplier for the WoS 360 path. Endpoints stay at the seed pose.")
    parser.add_argument("--wos_360_vertical_amp", type=float, default=0.04,
                        help="Vertical sinusoid amplitude as a fraction of center depth for the WoS 360 path.")
    parser.add_argument("--wos_360_clearance_ratio", type=float, default=0.06,
                        help="Minimum camera clearance from DA3 point cloud as a fraction of center depth.")
    parser.add_argument("--wos_360_step_fraction", type=float, default=0.9,
                        help="Fraction of the estimated free-space sphere used by each WoS step.")
    parser.add_argument("--wos_360_depth_stride", type=int, default=8,
                        help="Pixel stride for the sparse depth point cloud used by WoS planning.")
    parser.add_argument("--wos_360_max_points", type=int, default=20000,
                        help="Maximum sparse depth points used by the WoS planner.")
    parser.add_argument("--wos_360_depth_percentile", type=float, default=95.0,
                        help="Depth percentile cap used when building the WoS point cloud.")
    parser.add_argument("--wos_lookaround_yaw_degrees", type=float, default=360.0,
                        help="Total yaw sweep for --trajectory_preset wos_lookaround.")
    parser.add_argument("--wos_lookaround_pitch_degrees", type=float, default=35.0,
                        help="Maximum up/down pitch angle for --trajectory_preset wos_lookaround.")
    parser.add_argument("--wos_lookaround_pitch_cycles", type=float, default=1.0,
                        help="Number of smooth up/down pitch oscillations during the look-around.")
    parser.add_argument("--wos_lookaround_translation_ratio", type=float, default=0.008,
                        help="Small camera drift as a fraction of center depth for WoS look-around.")
    parser.add_argument("--walk_control_mode", type=str, default="random",
                        choices=["random", "keyboard", "file", "free_roam"],
                        help="Control source for --trajectory_preset wos_walk. "
                             "Use free_roam with the JSON written by free_roam_control.py.")
    parser.add_argument("--walk_control_path", type=str, default=None,
                        help="Optional JSON command file for --walk_control_mode file/free_roam.")
    parser.add_argument("--walk_speed_ratio", type=float, default=0.00025,
                        help="Forward step per generated frame as a fraction of center depth.")
    parser.add_argument("--walk_turn_degrees_per_chunk", type=float, default=18.0,
                        help="Maximum turn applied over one AR chunk.")
    parser.add_argument("--walk_random_turn_degrees", type=float, default=12.0,
                        help="Random turn noise for autonomous walking mode.")
    parser.add_argument("--walk_strafe_ratio", type=float, default=0.0015,
                        help="Lateral walking sway as a fraction of center depth.")
    parser.add_argument("--walk_bob_ratio", type=float, default=0.0015,
                        help="Vertical camera bob as a fraction of center depth.")
    parser.add_argument("--walk_pitch_degrees", type=float, default=5.0,
                        help="Maximum pitch command used by walking trajectory control.")
    parser.add_argument("--walk_vertical_ratio", type=float, default=0.00020,
                        help="Free-roam vertical step per generated frame as a fraction of center depth.")
    parser.add_argument("--walk_free_roam_max_chunk_ratio", type=float, default=0.05,
                        help="Maximum distance a free-roam target may pull one generated chunk, "
                             "as a fraction of center depth. Prevents huge jumps while Lyra is sampling.")
    parser.add_argument("--walk_map_update_max_points", type=int, default=20000,
                        help="Maximum sparse map points fed back into the walking planner.")
    parser.add_argument("--resolution", type=str, default="480,832", help="H,W")
    parser.add_argument("--context_parallel_size", type=int, default=1)
    parser.add_argument("--lora_paths", type=str, default=None, nargs="+")
    parser.add_argument("--lora_weights", type=float, default=None, nargs="+")
    parser.add_argument("--offload", action="store_true")
    parser.add_argument("--offload_when_prompt", action="store_true")
    parser.add_argument(
        "--low_vram",
        action="store_true",
        help="Enable conservative memory defaults: model/VAE offload, prompt offload, DA3 offload, and smaller warp chunks.",
    )
    parser.add_argument("--debug", action="store_true")

    # Depth backend
    parser.add_argument("--use_moge_scale", action=argparse.BooleanOptionalAction, default=True,
                        help="Align DA3 depth to MoGe scale (default: True).")
    parser.add_argument("--depth_backend", type=str, default="da3", choices=["da3"])
    parser.add_argument("--da3_model_name", type=str, default="depth-anything/DA3NESTED-GIANT-LARGE-1.1")
    parser.add_argument("--da3_model_path_custom", type=str, default=None)
    parser.add_argument("--da3_frame_interval", type=int, default=8)
    parser.add_argument("--da3_max_history_frames", type=int, default=10)
    parser.add_argument("--da3_include_ar_chunk_last_frames", action="store_true")
    parser.add_argument("--da3_use_predicted_pose", action="store_true")
    parser.add_argument("--da3_predicted_pose_continuation", action="store_true")

    # DMD distillation (4-step fast inference)
    parser.add_argument("--use_dmd", action="store_true",
                        help="Enable DMD fast inference: loads DMD distillation LoRA, "
                             "activates DMD scheduler, and reduces sampling steps.")

    # Misc flags needed by run_lyra2_sample internals
    parser.add_argument("--ablate_same_t5", action="store_true")
    parser.add_argument("--use_dmd_scheduler", action="store_true")
    parser.add_argument("--warp_chunk_size", type=int, default=None)
    parser.add_argument("--num_retrieval_views", type=int, default=1)
    parser.add_argument("--disable_cache_update", action="store_true")
    parser.add_argument("--multiview_ids", type=int, nargs="+", default=None)
    parser.add_argument("--offload_da3_diffusion", action="store_true")
    parser.add_argument(
        "--save_warp_video",
        action="store_true",
        help="Save/debug warped conditioning videos. Disabled by default to reduce memory and host transfers.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DMD_LORA_PATH = "checkpoints/lora/dmd_distillation.safetensors"
DMD_LORA_WEIGHT = 1.0


def _apply_dmd_defaults(args):
    """When --use_dmd is set, inject the DMD LoRA and switch to the DMD scheduler.

    Note: the DMD scheduler uses a fixed 4-step denoising list internally, so
    ``--num_sampling_step`` is ignored in this code path.
    """
    if not args.use_dmd:
        return
    args.use_dmd_scheduler = True
    if args.lora_paths is None:
        args.lora_paths = []
    if args.lora_weights is None:
        args.lora_weights = []
    args.lora_paths.append(DMD_LORA_PATH)
    args.lora_weights.append(DMD_LORA_WEIGHT)
    log.info(
        f"[DMD] Enabled: lora={DMD_LORA_PATH}, scheduler=dmd (4 fixed steps)",
        rank0_only=True,
    )


def _apply_low_vram_defaults(args):
    if not getattr(args, "low_vram", False):
        return
    args.offload = True
    args.offload_when_prompt = True
    args.offload_da3_diffusion = True
    if args.warp_chunk_size is None:
        args.warp_chunk_size = 2
    log.info(
        "[low_vram] Enabled: offload=True, offload_when_prompt=True, "
        f"offload_da3_diffusion=True, warp_chunk_size={args.warp_chunk_size}",
        rank0_only=True,
    )


if __name__ == "__main__":
    args = parse_arguments()
    _apply_low_vram_defaults(args)
    _apply_dmd_defaults(args)

    if args.debug:
        import debugpy
        debugpy.listen(5678)
        log.info("Waiting for debugger to attach...")
        debugpy.wait_for_client()

    process_group = None
    if args.context_parallel_size > 1:
        import imaginaire
        from megatron.core import parallel_state
        imaginaire.utils.distributed.init()
        parallel_state.initialize_model_parallel(context_parallel_size=args.context_parallel_size)
        process_group = parallel_state.get_context_parallel_group()

    os.makedirs(args.output_path, exist_ok=True)
    misc.set_random_seed(seed=args.seed, by_rank=True)

    # Negative prompt embeddings
    negative_prompt_data = torch.load(
        "checkpoints/text_encoder/negative_prompt.pt", map_location="cpu", weights_only=False
    )

    # ---- Load FramePack model ----
    experiment_opts = [
        "model.config.use_mp_policy_fsdp=False",
        "model.config.keep_original_net_dtype=False",
    ]
    if args.lora_paths:
        experiment_opts += ["model.config.net.postpone_checkpoint=True"]
    model, config = load_model_from_checkpoint(
        config_file="lyra_2/_src/configs/config.py",
        experiment_name=args.experiment,
        checkpoint_path=args.checkpoint_dir,
        enable_fsdp=False,
        instantiate_ema=False,
        load_ema_to_reg=False,
        experiment_opts=experiment_opts,
    )
    if args.lora_paths:
        lora_names = []
        for lora_path in args.lora_paths:
            lora_name = model.load_lora_weights(lora_path)
            lora_names.append(lora_name)
        model.set_weights_and_activate_adapters(lora_names, args.lora_weights)
        if args.low_vram and hasattr(model, "merge_active_lora_adapters"):
            model.merge_active_lora_adapters(lora_names)
        if args.low_vram:
            log.info("Skipping selective checkpoint wrappers for low-VRAM inference", rank0_only=True)
        elif hasattr(model, "net") and hasattr(model.net, "enable_selective_checkpoint"):
            model.net.enable_selective_checkpoint(model.net.sac_config, model.net.blocks)

    desired_dtype = model.tensor_kwargs.get("dtype", None)
    desired_device = model.tensor_kwargs.get("device", None)
    if desired_dtype is not None:
        model.net = model.net.to(device=desired_device, dtype=desired_dtype)
        log.info(f"Casted model.net to dtype={desired_dtype}", rank0_only=True)

    assert getattr(model.config, "important_start", True) is True
    assert getattr(model.config, "encode_video_from_start", True) is True
    assert not getattr(model.config, "use_hd_map_cond", False)

    model.eval()
    if args.context_parallel_size > 1:
        model.net.enable_context_parallel(process_group)

    if args.warp_chunk_size is not None:
        model.config.warp_chunk_size = args.warp_chunk_size
        model.warp_chunk_size = args.warp_chunk_size

    if args.trajectory_preset == "wos_walk" and int(args.continuous_chunks) <= 0 and not args.unbounded_stream:
        args.continuous_chunks = max(1, (int(args.num_frames) - 1) // int(model.framepack_num_new_video_frames))

    if int(args.continuous_chunks) > 0:
        args.num_frames = 1 + int(args.continuous_chunks) * int(model.framepack_num_new_video_frames)
        args.stream_output_chunks = True
        args.enable_history_ring = True
        if args.trajectory_preset == "file" and args.trajectory_extend == "error":
            args.trajectory_extend = "loop"
        if args.trajectory_preset == "wos_lookaround":
            yaw_per_chunk = abs(float(args.wos_lookaround_yaw_degrees)) / max(1, int(args.continuous_chunks))
            if yaw_per_chunk > 45.0:
                recommended_chunks = int(math.ceil(abs(float(args.wos_lookaround_yaw_degrees)) / 35.0))
                log.warning(
                    f"[wos_lookaround] yaw_per_chunk={yaw_per_chunk:.1f} deg is likely too fast. "
                    f"For steadier 360 look-around, use --continuous_chunks {recommended_chunks} "
                    "or reduce --wos_lookaround_yaw_degrees.",
                    rank0_only=True,
                )
        log.info(
            f"[continuous] chunks={args.continuous_chunks}, num_frames={args.num_frames}, "
            f"trajectory_extend={args.trajectory_extend}, ring_cache_entries={args.ring_cache_entries}",
            rank0_only=True,
        )
    elif args.unbounded_stream:
        args.stream_output_chunks = True
        args.enable_history_ring = True
        log.info(
            f"[continuous] unbounded stream enabled, stream_max_chunks={args.stream_max_chunks}, "
            f"ring_cache_entries={args.ring_cache_entries}",
            rank0_only=True,
        )
    else:
        args.enable_history_ring = bool(args.stream_output_chunks)

    # Resolution
    target_h, target_w = [int(x) for x in args.resolution.split(",")]

    # ---- Load DA3 model ----
    from lyra_2._src.inference.depth_utils import load_da3_model
    da3_target_device = model.tensor_kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    da3_device = "cpu" if args.offload_da3_diffusion else da3_target_device
    da3_model = load_da3_model(
        da3_model_name=args.da3_model_name,
        da3_model_path_custom=args.da3_model_path_custom,
        device=da3_device,
    )
    da3_model.eval()

    # ---- Optionally load MoGe model for depth scale alignment ----
    moge_model = None
    if args.use_moge_scale:
        from lyra_2._src.inference.depth_utils import load_moge_model
        moge_device = model.tensor_kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        moge_model = load_moge_model(moge_device)
        moge_model.eval()
        log.info("MoGe model loaded for depth scale alignment.", rank0_only=True)

    # ---- Resolve image(s) ----
    image_paths = _build_image_list(args.input_image_path)[
        args.sample_start_idx : args.sample_start_idx + args.num_samples
    ]

    if args.trajectory_preset == "file" and args.trajectory_path is None:
        raise ValueError("--trajectory_path is required when --trajectory_preset=file")

    # Resolve trajectory file(s): single file shared across images, or per-image files in a folder.
    traj_is_dir = args.trajectory_path is not None and os.path.isdir(args.trajectory_path)

    # Resolve captions source: per-chunk JSON or single caption
    captions_is_dir = args.captions_path is not None and os.path.isdir(args.captions_path)

    N = int(args.num_frames)

    for img_idx, img_path in enumerate(image_paths):
        base_name = os.path.splitext(os.path.basename(img_path))[0]

        video_path = os.path.join(args.output_path, f"{base_name}.mp4")
        if args.stream_output_chunks:
            args.stream_output_dir = os.path.join(args.output_path, f"{base_name}_stream")
            if args.stream_preview_frames and args.stream_preview_dir is None:
                args.stream_preview_dir = os.path.join(args.stream_output_dir, "preview")
            stream_manifest_path = os.path.join(args.stream_output_dir, "concat.txt")
            video_exists = os.path.isdir(args.stream_output_dir) and os.path.exists(
                stream_manifest_path
            ) and os.path.getsize(
                stream_manifest_path
            )
        else:
            args.stream_output_dir = None
            if args.stream_preview_frames and args.stream_preview_dir is None:
                args.stream_preview_dir = None
            video_exists = os.path.exists(video_path)
        if video_exists:
            log.info(f"Skipping {img_path} (video already exists at {video_path})", rank0_only=True)
            continue

        log.info(f"Processing [{img_idx}]: {img_path}", rank0_only=True)
        misc.set_random_seed(seed=args.seed, by_rank=True)

        # ---- Read image ----
        bgr = cv2.imread(img_path)
        if bgr is None:
            log.error(f"Cannot read: {img_path}")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_t = torch.from_numpy(rgb)

        # ---- Depth & intrinsics for the first frame (via DA3) ----
        log.info("Running DA3 single-image depth...", rank0_only=True)
        if args.offload_da3_diffusion:
            _offload_diffusion_to_cpu(model, True)
            _restore_module_to_device(da3_model, True, da3_target_device)
        try:
            image_chw01, depth_hw, K_33_da3, mask_hw = _da3_infer_depth_intrinsics_single(
                da3_model=da3_model,
                img_rgb_uint8=rgb_t,
                target_hw=(target_h, target_w),
            )
        finally:
            if args.offload_da3_diffusion:
                _offload_module_to_cpu(da3_model, True)
                _restore_diffusion_to_gpu(model, True)
        H, W = image_chw01.shape[-2:]

        # ---- Optionally align DA3 depth to MoGe scale ----
        if args.use_moge_scale and moge_model is not None:
            log.info("Aligning DA3 depth to MoGe scale...", rank0_only=True)
            from lyra_2._src.inference.depth_utils import moge_infer_depth_intrinsics

            if args.offload_da3_diffusion:
                _offload_diffusion_to_cpu(model, True)
            try:
                moge_model.to(desired_device)
                with torch.nn.attention.sdpa_kernel(
                    [torch.nn.attention.SDPBackend.MATH]
                ):
                    _, moge_depth_hw, _, moge_mask_hw = moge_infer_depth_intrinsics(
                        moge_model,
                        rgb_t,
                        depth_pred_hw=(target_h, target_w),
                        target_hw=(target_h, target_w),
                    )
            finally:
                moge_model.cpu()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if args.offload_da3_diffusion:
                    _restore_diffusion_to_gpu(model, True)

            da3_d = depth_hw.to(moge_depth_hw.device)
            da3_m = mask_hw.to(moge_mask_hw.device)

            valid_mask = (da3_m > 0.5) & (moge_mask_hw > 0.5)
            if valid_mask.sum() > 10:
                d_da3_vals = da3_d[valid_mask]
                d_moge_vals = moge_depth_hw[valid_mask]

                inv_da3 = 1.0 / (d_da3_vals + 1e-6)
                inv_moge = 1.0 / (d_moge_vals + 1e-6)

                numerator = (inv_da3 * inv_moge).sum()
                denominator = (inv_da3 * inv_da3).sum()

                if denominator > 1e-8:
                    scale = numerator / denominator
                    log.info(f"Global inverse-depth scale factor: {scale.item()}", rank0_only=True)
                    if scale > 1e-6:
                        depth_hw = depth_hw / scale.to(depth_hw.device)
                    else:
                        log.warning(f"Scale too small ({scale.item()}), skipping alignment.", rank0_only=True)
                else:
                    log.warning("Denominator too small for LS scale alignment.", rank0_only=True)
            else:
                log.warning("Not enough overlapping valid pixels for scale alignment.", rank0_only=True)

            del moge_depth_hw, moge_mask_hw, da3_d, da3_m
            torch.cuda.empty_cache()
            gc.collect()

        # ---- Load or generate trajectory ----
        trajectory_stream = None
        if args.trajectory_preset == "wos_360":
            w2cs_T_44, Ks_T_33, wos_stats = build_wos_360_trajectory(
                depth_hw,
                K_33_da3,
                mask_hw,
                num_frames=N,
                loops=args.wos_360_loops,
                radius_scale=args.wos_360_radius_scale,
                vertical_amp=args.wos_360_vertical_amp,
                clearance_ratio=args.wos_360_clearance_ratio,
                step_fraction=args.wos_360_step_fraction,
                depth_stride=args.wos_360_depth_stride,
                max_points=args.wos_360_max_points,
                depth_percentile=args.wos_360_depth_percentile,
            )
            traj_file = os.path.join(args.output_path, f"{base_name}_wos_360_trajectory.npz")
            np.savez(
                traj_file,
                w2c=w2cs_T_44.numpy().astype(np.float32),
                intrinsics=Ks_T_33.numpy().astype(np.float32),
                image_height=np.array(target_h, dtype=np.int64),
                image_width=np.array(target_w, dtype=np.int64),
                center_depth=np.array(wos_stats["center_depth"], dtype=np.float32),
                clearance=np.array(wos_stats["clearance"], dtype=np.float32),
                sparse_points=np.array(wos_stats["num_points"], dtype=np.int64),
            )
            log.info(
                f"Generated WoS 360 trajectory: {traj_file} "
                f"(frames={w2cs_T_44.shape[0]}, center_depth={wos_stats['center_depth']:.4f}, "
                f"clearance={wos_stats['clearance']:.4f}, points={int(wos_stats['num_points'])})",
                rank0_only=True,
            )
        elif args.trajectory_preset == "wos_walk":
            trajectory_stream = WosWalkTrajectoryStream(
                depth_hw,
                K_33_da3,
                mask_hw,
                seed=args.seed,
                speed_ratio=args.walk_speed_ratio,
                turn_degrees_per_chunk=args.walk_turn_degrees_per_chunk,
                random_turn_degrees=args.walk_random_turn_degrees,
                strafe_ratio=args.walk_strafe_ratio,
                bob_ratio=args.walk_bob_ratio,
                pitch_degrees=args.walk_pitch_degrees,
                vertical_ratio=args.walk_vertical_ratio,
                free_roam_max_chunk_ratio=args.walk_free_roam_max_chunk_ratio,
                clearance_ratio=args.wos_360_clearance_ratio,
                step_fraction=args.wos_360_step_fraction,
                depth_stride=args.wos_360_depth_stride,
                max_points=args.wos_360_max_points,
                depth_percentile=args.wos_360_depth_percentile,
                control_mode=args.walk_control_mode,
                control_path=args.walk_control_path,
                map_update_max_points=args.walk_map_update_max_points,
            )
            w2cs_T_44 = torch.eye(4, dtype=torch.float32).unsqueeze(0)
            Ks_T_33 = K_33_da3.detach().to(dtype=torch.float32, device="cpu").unsqueeze(0)
            traj_file = os.path.join(args.output_path, f"{base_name}_wos_walk_seed_trajectory.npz")
            np.savez(
                traj_file,
                w2c=w2cs_T_44.numpy().astype(np.float32),
                intrinsics=Ks_T_33.numpy().astype(np.float32),
                image_height=np.array(target_h, dtype=np.int64),
                image_width=np.array(target_w, dtype=np.int64),
                center_depth=np.array(trajectory_stream.center_depth, dtype=np.float32),
                clearance=np.array(trajectory_stream.clearance, dtype=np.float32),
                sparse_points=np.array(trajectory_stream.points.shape[0], dtype=np.int64),
                control_mode=np.array(args.walk_control_mode),
            )
            log.info(
                f"Initialized WoS walk stream: {traj_file} "
                f"(center_depth={trajectory_stream.center_depth:.4f}, "
                f"clearance={trajectory_stream.clearance:.4f}, points={trajectory_stream.points.shape[0]}, "
                f"control={args.walk_control_mode})",
                rank0_only=True,
            )
        elif args.trajectory_preset == "wos_lookaround":
            w2cs_T_44, Ks_T_33, wos_stats = build_wos_lookaround_trajectory(
                depth_hw,
                K_33_da3,
                mask_hw,
                num_frames=N,
                yaw_degrees=args.wos_lookaround_yaw_degrees,
                pitch_degrees=args.wos_lookaround_pitch_degrees,
                pitch_cycles=args.wos_lookaround_pitch_cycles,
                translation_ratio=args.wos_lookaround_translation_ratio,
                clearance_ratio=args.wos_360_clearance_ratio,
                step_fraction=args.wos_360_step_fraction,
                depth_stride=args.wos_360_depth_stride,
                max_points=args.wos_360_max_points,
                depth_percentile=args.wos_360_depth_percentile,
            )
            traj_file = os.path.join(args.output_path, f"{base_name}_wos_lookaround_trajectory.npz")
            np.savez(
                traj_file,
                w2c=w2cs_T_44.numpy().astype(np.float32),
                intrinsics=Ks_T_33.numpy().astype(np.float32),
                image_height=np.array(target_h, dtype=np.int64),
                image_width=np.array(target_w, dtype=np.int64),
                center_depth=np.array(wos_stats["center_depth"], dtype=np.float32),
                clearance=np.array(wos_stats["clearance"], dtype=np.float32),
                sparse_points=np.array(wos_stats["num_points"], dtype=np.int64),
                yaw_degrees=np.array(wos_stats["yaw_degrees"], dtype=np.float32),
                pitch_degrees=np.array(wos_stats["pitch_degrees"], dtype=np.float32),
                max_drift=np.array(wos_stats["max_drift"], dtype=np.float32),
            )
            log.info(
                f"Generated WoS look-around trajectory: {traj_file} "
                f"(frames={w2cs_T_44.shape[0]}, yaw={wos_stats['yaw_degrees']:.1f}, "
                f"pitch=+/-{wos_stats['pitch_degrees']:.1f}, max_drift={wos_stats['max_drift']:.4f}, "
                f"clearance={wos_stats['clearance']:.4f}, points={int(wos_stats['num_points'])})",
                rank0_only=True,
            )
        else:
            if traj_is_dir:
                traj_file = os.path.join(args.trajectory_path, f"{base_name}.npz")
            else:
                traj_file = args.trajectory_path
            if not os.path.isfile(traj_file):
                log.error(f"Trajectory file not found: {traj_file}")
                continue

            w2cs_T_44, Ks_T_33 = load_trajectory(
                traj_file,
                N,
                target_hw=(target_h, target_w),
                pose_scale=args.pose_scale,
                extend_mode=args.trajectory_extend,
            )
            log.info(f"Loaded trajectory: {w2cs_T_44.shape[0]} frames from {traj_file}", rank0_only=True)

        img_bchw = image_chw01.to(device=desired_device) * 2.0 - 1.0

        # ---- Load captions ----
        neg_t5 = misc.to(negative_prompt_data["t5_text_embeddings"], **model.tensor_kwargs)

        captions_file = None
        if args.captions_path is not None:
            if captions_is_dir:
                captions_file = os.path.join(args.captions_path, f"{base_name}.json")
            else:
                captions_file = args.captions_path
            if not os.path.isfile(captions_file):
                log.warning(f"Captions file not found: {captions_file}, falling back to single caption")
                captions_file = None

        use_chunk_captions = False
        if captions_file is not None:
            with open(captions_file, "r") as f:
                captions_dict = json.load(f)
            chunk_keys_int = sorted(int(k) for k in captions_dict)
            chunk_keys_int = [k for k in chunk_keys_int if k < N]
            if len(chunk_keys_int) > 1:
                use_chunk_captions = True
                log.info(f"Loaded {len(chunk_keys_int)} chunk captions from {captions_file}", rank0_only=True)

                chunk_keys = torch.tensor(chunk_keys_int, dtype=torch.long, device=desired_device)
                chunk_embs = []
                chunk_masks = []
                for ck in chunk_keys_int:
                    cap = captions_dict[str(ck)]
                    if args.prompt_suffix:
                        cap = cap.rstrip() + " " + args.prompt_suffix
                    emb = _get_t5_embedding_memory_safe(cap, args, desired_device, desired_dtype, model)
                    if emb.dim() == 3:
                        emb = emb[0]
                    S, D = emb.shape
                    S = min(S, 512)
                    D = min(D, 4096)
                    padded_emb = torch.zeros(512, 4096, dtype=desired_dtype, device=desired_device)
                    padded_emb[:S, :D] = emb[:S, :D]
                    padded_mask = torch.zeros(512, dtype=desired_dtype, device=desired_device)
                    padded_mask[:S] = 1.0
                    chunk_embs.append(padded_emb)
                    chunk_masks.append(padded_mask)

                t5_chunk_embeddings = torch.stack(chunk_embs).unsqueeze(0)
                t5_chunk_mask = torch.stack(chunk_masks).unsqueeze(0)
                t5_chunk_keys = chunk_keys.unsqueeze(0)
                sample_frame_indices = torch.arange(N, dtype=torch.long, device=desired_device).unsqueeze(0)
                t5 = t5_chunk_embeddings[:, 0, :, :]
            else:
                single_caption = captions_dict.get(str(chunk_keys_int[0]), "") if chunk_keys_int else ""
                if args.prompt_suffix:
                    single_caption = single_caption.rstrip() + " " + args.prompt_suffix

        if not use_chunk_captions:
            if args.prompt:
                caption = args.prompt
            elif captions_file is not None:
                caption = single_caption
            elif args.prompt_dir:
                txt_path = os.path.join(args.prompt_dir, f"{base_name}.txt")
                if not os.path.isfile(txt_path):
                    log.error(f"Caption file not found: {txt_path}")
                    continue
                with open(txt_path, "r") as f:
                    caption = f.read().strip()
                log.info(f"Loaded caption from {txt_path}", rank0_only=True)
            else:
                raise RuntimeError(
                    "No caption source specified. Use --captions_path, --prompt, or --prompt_dir."
                )
            if args.prompt_suffix:
                caption = caption.rstrip() + " " + args.prompt_suffix
            t5 = _get_t5_embedding_memory_safe(caption, args, desired_device, desired_dtype, model)
            if t5.dim() == 2:
                t5 = t5.unsqueeze(0)
            elif t5.dim() == 3 and t5.shape[0] != 1:
                t5 = t5[:1]

        # ---- Assemble data batch ----
        w2cs_b_t_44 = w2cs_T_44.unsqueeze(0).to(dtype=torch.float32, device=desired_device)
        Ks_b_t_33 = Ks_T_33.unsqueeze(0).to(dtype=torch.float32, device=desired_device)
        depth_T = 1 if trajectory_stream is not None else N
        depth_b_thw = depth_hw.unsqueeze(0).unsqueeze(0).repeat(1, depth_T, 1, 1).to(device=desired_device)

        data_batch = {
            "video": img_bchw.unsqueeze(2),
            "t5_text_embeddings": t5,
            "neg_t5_text_embeddings": neg_t5,
            "fps": torch.tensor([args.fps], dtype=torch.int32, device=desired_device),
            "padding_mask": torch.zeros((1, 1, H, W), dtype=model.tensor_kwargs["dtype"], device=desired_device),
            "is_preprocessed": torch.tensor([True], dtype=torch.bool, device=desired_device),
            "camera_w2c": w2cs_b_t_44,
            "intrinsics": Ks_b_t_33,
            "depth": depth_b_thw,
        }
        if trajectory_stream is not None:
            data_batch["trajectory_stream"] = trajectory_stream

        if use_chunk_captions:
            data_batch["t5_chunk_keys"] = t5_chunk_keys
            data_batch["t5_chunk_embeddings"] = t5_chunk_embeddings
            data_batch["t5_chunk_mask"] = t5_chunk_mask
            data_batch["sample_frame_indices"] = sample_frame_indices

        skip_keys = {"camera_w2c", "intrinsics", "depth", "t5_chunk_keys", "sample_frame_indices"}
        data_batch = safe_to(
            data_batch,
            device=model.tensor_kwargs.get("device", None),
            dtype=model.tensor_kwargs.get("dtype", None),
            skip_keys=skip_keys,
        )

        # ---- Run AR inference ----
        if trajectory_stream is not None and args.unbounded_stream:
            log.info("=== Generating unbounded streaming video ===", rank0_only=True)
        else:
            log.info(f"=== Generating video ({N} frames) ===", rank0_only=True)
        result = run_lyra2_sample(
            model,
            data_batch,
            args,
            process_group=process_group,
            da3_model=da3_model,
            show_progress=True,
            log_prefix=f"{base_name}_custom_traj",
        )

        if result is None:
            log.warning(f"Generation failed for {img_path}", rank0_only=True)
            continue

        if args.stream_output_chunks:
            log.info(
                f"Saved streaming chunks: {result['stream_dir']} (manifest: {result['manifest']})",
                rank0_only=True,
            )
            del result, data_batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            continue

        # ---- Save output video ----
        video_01 = (result["video"][0].clamp(-1, 1) * 0.5 + 0.5).float().cpu()
        save_img_or_video(video_01, video_path.replace(".mp4", ""), fps=args.fps)
        log.info(f"Saved video: {video_path}", rank0_only=True)

        del result, data_batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # Clean up distributed
    if args.context_parallel_size > 1:
        from megatron.core import parallel_state
        parallel_state.destroy_model_parallel()
        try:
            import torch.distributed as dist
            dist.destroy_process_group()
        except Exception:
            pass

    log.info("Done.", rank0_only=True)
