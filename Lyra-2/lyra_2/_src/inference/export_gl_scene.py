# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export a Lyra2 seed-view reconstruction and camera path as GL-viewable assets.

The exporter writes:
  * scene.ply  - colored point cloud
  * scene.glb  - point cloud plus camera path/frustums
  * scene.usda - USD scene with points and frustum/path curves

The point cloud is reconstructed from the first image and either a provided depth
map or DA3 single-image depth. The camera path is read from a Lyra trajectory
``.npz`` with ``w2c`` and ``intrinsics`` arrays.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch


def _load_rgb(path: str, target_hw: tuple[int, int]) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    H, W = target_hw
    if rgb.shape[:2] != (H, W):
        rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)
    return rgb


def _load_depth(path: str, target_hw: tuple[int, int]) -> np.ndarray:
    p = Path(path)
    if p.suffix == ".npy":
        depth = np.load(path)
    elif p.suffix == ".npz":
        data = np.load(path)
        for key in ("depth", "depth_hw", "depths"):
            if key in data:
                depth = data[key]
                break
        else:
            raise KeyError(f"No depth/depth_hw/depths key found in {path}")
    else:
        raise ValueError(f"Unsupported depth format: {path}")

    depth = np.asarray(depth, dtype=np.float32)
    while depth.ndim > 2:
        depth = depth[0]
    H, W = target_hw
    if depth.shape != (H, W):
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)
    return np.nan_to_num(depth, nan=1e4).astype(np.float32)


def _infer_da3_depth(
    image_path: str,
    target_hw: tuple[int, int],
    *,
    da3_model_name: str,
    da3_model_path_custom: str | None,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    from lyra_2._src.inference.depth_utils import load_da3_model
    from lyra_2._src.inference.lyra2_zoomgs_inference import _da3_infer_depth_intrinsics_single

    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb_t = torch.from_numpy(rgb)

    da3_model = load_da3_model(
        da3_model_name=da3_model_name,
        da3_model_path_custom=da3_model_path_custom,
        device=device,
    )
    da3_model.eval()
    with torch.no_grad():
        _img, depth_hw, K_33, _mask = _da3_infer_depth_intrinsics_single(
            da3_model=da3_model,
            img_rgb_uint8=rgb_t,
            target_hw=target_hw,
        )
    return depth_hw.detach().cpu().numpy().astype(np.float32), K_33.detach().cpu().numpy().astype(np.float32)


def _trajectory_hw(data: np.lib.npyio.NpzFile, fallback_hw: tuple[int, int] | None) -> tuple[int, int]:
    if "image_height" in data and "image_width" in data:
        return int(data["image_height"]), int(data["image_width"])
    if fallback_hw is None:
        raise ValueError("Trajectory has no image_height/image_width; pass --resolution H,W.")
    return fallback_hw


def _select_point_indices(valid: np.ndarray, stride: int, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    H, W = valid.shape
    yy, xx = np.mgrid[0:H:max(1, stride), 0:W:max(1, stride)]
    yy = yy.reshape(-1)
    xx = xx.reshape(-1)
    keep = valid[yy, xx]
    yy = yy[keep]
    xx = xx[keep]
    if yy.size == 0:
        yy, xx = np.nonzero(valid)
    if max_points > 0 and yy.size > max_points:
        sel = np.linspace(0, yy.size - 1, max_points, dtype=np.int64)
        yy = yy[sel]
        xx = xx[sel]
    return yy, xx


def _backproject_colored_points(
    rgb: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    first_w2c: np.ndarray,
    *,
    stride: int,
    max_points: int,
    depth_percentile: float,
) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(depth) & (depth > 1e-4) & (depth < 1e4)
    if not np.any(finite):
        raise ValueError("Depth has no valid pixels.")
    cap = np.percentile(depth[finite], float(depth_percentile))
    valid = finite & (depth <= cap)
    yy, xx = _select_point_indices(valid, int(stride), int(max_points))

    z = depth[yy, xx].astype(np.float32)
    x = ((xx.astype(np.float32) - float(K[0, 2])) / max(float(K[0, 0]), 1e-6)) * z
    y = ((yy.astype(np.float32) - float(K[1, 2])) / max(float(K[1, 1]), 1e-6)) * z
    cam_pts = np.stack([x, y, z, np.ones_like(z)], axis=1)
    c2w0 = np.linalg.inv(first_w2c.astype(np.float32))
    world = (c2w0 @ cam_pts.T).T[:, :3].astype(np.float32)
    colors = rgb[yy, xx].astype(np.uint8)
    return world, colors


def _camera_center(w2c: np.ndarray) -> np.ndarray:
    c2w = np.linalg.inv(w2c.astype(np.float32))
    return c2w[:3, 3].astype(np.float32)


def _frustum_segments(
    w2c: np.ndarray,
    K: np.ndarray,
    image_hw: tuple[int, int],
    depth: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    H, W = image_hw
    pix = np.array([[0, 0], [W, 0], [W, H], [0, H]], dtype=np.float32)
    z = float(depth)
    corners_cam = np.stack(
        [
            (pix[:, 0] - float(K[0, 2])) / max(float(K[0, 0]), 1e-6) * z,
            (pix[:, 1] - float(K[1, 2])) / max(float(K[1, 1]), 1e-6) * z,
            np.full(4, z, dtype=np.float32),
            np.ones(4, dtype=np.float32),
        ],
        axis=1,
    )
    c2w = np.linalg.inv(w2c.astype(np.float32))
    center = c2w[:3, 3].astype(np.float32)
    corners = (c2w @ corners_cam.T).T[:, :3].astype(np.float32)
    segs = [(center, c) for c in corners]
    segs.extend((corners[i], corners[(i + 1) % 4]) for i in range(4))
    return segs


def _path_segments(points: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    if len(points) < 2:
        return []
    return [(points[i], points[i + 1]) for i in range(len(points) - 1)]


def _write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    from plyfile import PlyData, PlyElement

    vertex = np.empty(
        len(points),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vertex["x"] = points[:, 0]
    vertex["y"] = points[:, 1]
    vertex["z"] = points[:, 2]
    vertex["red"] = colors[:, 0]
    vertex["green"] = colors[:, 1]
    vertex["blue"] = colors[:, 2]
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(str(path))


def _cylinder_between(a: np.ndarray, b: np.ndarray, radius: float, color: tuple[int, int, int, int]):
    import trimesh

    vec = np.asarray(b, dtype=np.float32) - np.asarray(a, dtype=np.float32)
    length = float(np.linalg.norm(vec))
    if length < 1e-6:
        return None
    cyl = trimesh.creation.cylinder(radius=radius, height=length, sections=8)
    transform = trimesh.geometry.align_vectors([0, 0, 1], vec / length)
    transform[:3, 3] = (np.asarray(a, dtype=np.float32) + np.asarray(b, dtype=np.float32)) * 0.5
    cyl.apply_transform(transform)
    cyl.visual.vertex_colors = np.tile(np.asarray(color, dtype=np.uint8), (len(cyl.vertices), 1))
    return cyl


def _write_glb(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    camera_centers: np.ndarray,
    segments: Iterable[tuple[np.ndarray, np.ndarray]],
    *,
    line_radius: float,
) -> None:
    import trimesh

    scene = trimesh.Scene()
    point_colors = np.concatenate([colors, np.full((len(colors), 1), 255, dtype=np.uint8)], axis=1)
    scene.add_geometry(trimesh.points.PointCloud(points, colors=point_colors), geom_name="seed_point_cloud")

    for idx, c in enumerate(camera_centers[:: max(1, len(camera_centers) // 24)]):
        sphere = trimesh.creation.uv_sphere(radius=line_radius * 4.0, count=[8, 8])
        sphere.apply_translation(c)
        sphere.visual.vertex_colors = np.tile(np.array([255, 180, 0, 255], dtype=np.uint8), (len(sphere.vertices), 1))
        scene.add_geometry(sphere, geom_name=f"camera_{idx:03d}")

    for idx, (a, b) in enumerate(segments):
        cyl = _cylinder_between(a, b, line_radius, (20, 180, 255, 255))
        if cyl is not None:
            scene.add_geometry(cyl, geom_name=f"line_{idx:05d}")

    scene.export(str(path))


def _vec3f_array(values: np.ndarray):
    from pxr import Gf, Vt

    return Vt.Vec3fArray([Gf.Vec3f(float(x), float(y), float(z)) for x, y, z in values])


def _write_usd(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    segments: Iterable[tuple[np.ndarray, np.ndarray]],
    *,
    point_width: float,
    line_width: float,
) -> None:
    from pxr import Gf, Usd, UsdGeom, Vt

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    usd_points = UsdGeom.Points.Define(stage, "/World/SeedPointCloud")
    usd_points.CreatePointsAttr(_vec3f_array(points))
    display_colors = colors.astype(np.float32) / 255.0
    usd_points.CreateDisplayColorAttr(_vec3f_array(display_colors))
    usd_points.CreateWidthsAttr(Vt.FloatArray([float(point_width)] * len(points)))

    line_pts: list[np.ndarray] = []
    counts: list[int] = []
    for a, b in segments:
        line_pts.append(np.asarray(a, dtype=np.float32))
        line_pts.append(np.asarray(b, dtype=np.float32))
        counts.append(2)

    if line_pts:
        curves = UsdGeom.BasisCurves.Define(stage, "/World/CameraPathAndFrustums")
        curves.CreateTypeAttr(UsdGeom.Tokens.linear)
        curves.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
        curves.CreateCurveVertexCountsAttr(Vt.IntArray(counts))
        curves.CreatePointsAttr(_vec3f_array(np.stack(line_pts, axis=0)))
        curves.CreateWidthsAttr(Vt.FloatArray([float(line_width)] * len(line_pts)))
        curves.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.05, 0.65, 1.0)]))

    stage.GetRootLayer().Save()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Lyra2 trajectory/depth as PLY, GLB, and USD scenes.")
    parser.add_argument("--input_image_path", required=True)
    parser.add_argument("--trajectory_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--depth_path", default=None, help="Optional .npy/.npz depth. If omitted, DA3 is run.")
    parser.add_argument("--resolution", default=None, help="Optional H,W override. Defaults to trajectory resolution.")
    parser.add_argument("--da3_model_name", default="depth-anything/DA3NESTED-GIANT-LARGE-1.1")
    parser.add_argument("--da3_model_path_custom", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_points", type=int, default=60000)
    parser.add_argument("--depth_stride", type=int, default=2)
    parser.add_argument("--depth_percentile", type=float, default=95.0)
    parser.add_argument("--camera_stride", type=int, default=16)
    parser.add_argument("--frustum_depth_ratio", type=float, default=0.12)
    parser.add_argument("--line_radius", type=float, default=0.01)
    parser.add_argument("--usd_point_width", type=float, default=0.02)
    parser.add_argument("--usd_line_width", type=float, default=0.015)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    traj = np.load(args.trajectory_path)
    fallback_hw = None
    if args.resolution:
        h, w = [int(x) for x in args.resolution.split(",")]
        fallback_hw = (h, w)
    target_hw = _trajectory_hw(traj, fallback_hw)

    w2c = traj["w2c"].astype(np.float32)
    K = traj["intrinsics"].astype(np.float32)
    rgb = _load_rgb(args.input_image_path, target_hw)

    if args.depth_path:
        depth = _load_depth(args.depth_path, target_hw)
    else:
        depth, K_da3 = _infer_da3_depth(
            args.input_image_path,
            target_hw,
            da3_model_name=args.da3_model_name,
            da3_model_path_custom=args.da3_model_path_custom,
            device=args.device,
        )
        np.save(out_dir / "seed_depth.npy", depth)
        if "intrinsics" not in traj:
            K[0] = K_da3

    points, colors = _backproject_colored_points(
        rgb,
        depth,
        K[0],
        w2c[0],
        stride=args.depth_stride,
        max_points=args.max_points,
        depth_percentile=args.depth_percentile,
    )

    centers = np.stack([_camera_center(m) for m in w2c], axis=0)
    center_depth = float(traj["center_depth"]) if "center_depth" in traj else float(np.nanmedian(depth))
    frustum_depth = max(center_depth * float(args.frustum_depth_ratio), 1e-3)
    segments = _path_segments(centers)
    for i in range(0, len(w2c), max(1, int(args.camera_stride))):
        segments.extend(_frustum_segments(w2c[i], K[min(i, len(K) - 1)], target_hw, frustum_depth))

    ply_path = out_dir / "scene.ply"
    glb_path = out_dir / "scene.glb"
    usd_path = out_dir / "scene.usda"
    _write_ply(ply_path, points, colors)
    _write_glb(glb_path, points, colors, centers, segments, line_radius=float(args.line_radius))
    _write_usd(
        usd_path,
        points,
        colors,
        segments,
        point_width=float(args.usd_point_width),
        line_width=float(args.usd_line_width),
    )

    metadata = {
        "input_image_path": os.path.abspath(args.input_image_path),
        "trajectory_path": os.path.abspath(args.trajectory_path),
        "output_dir": os.path.abspath(str(out_dir)),
        "target_hw": list(target_hw),
        "num_points": int(len(points)),
        "num_cameras": int(len(w2c)),
        "num_segments": int(len(segments)),
        "center_depth": center_depth,
        "files": {
            "ply": str(ply_path),
            "glb": str(glb_path),
            "usd": str(usd_path),
        },
    }
    with open(out_dir / "scene_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
