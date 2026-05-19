# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Newton-style GL free-roam controller for Lyra2 streaming trajectories.

This is a small input window, not a renderer for generated video. It mirrors the
camera controls used by Newton's GL viewer: WASD movement, QE vertical movement,
arrow-key or left-mouse look, and smooth velocity damping. Every update writes a
JSON command file that ``lyra2_custom_traj_inference.py --walk_control_mode
free_roam`` can follow chunk by chunk.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def _parse_hw(value: str) -> tuple[int, int]:
    parts = [int(item.strip()) for item in value.replace("x", ",").split(",") if item.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected H,W, for example 256,448")
    return parts[0], parts[1]


def _parse_vec3(value: str) -> np.ndarray:
    parts = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected x,y,z")
    return np.asarray(parts, dtype=np.float32)


def _short_angle_delta_degrees(target: float, current: float) -> float:
    return (float(target) - float(current) + 180.0) % 360.0 - 180.0


def _look_at_w2c(camera_pos: np.ndarray, target: np.ndarray) -> np.ndarray:
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


def _intrinsics_from_fov(image_hw: tuple[int, int], fov_degrees: float) -> np.ndarray:
    h, w = image_hw
    fy = 0.5 * float(h) / max(math.tan(math.radians(float(fov_degrees)) * 0.5), 1e-6)
    fx = fy
    return np.asarray(
        [
            [fx, 0.0, 0.5 * float(w)],
            [0.0, fy, 0.5 * float(h)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


@dataclass
class FreeRoamCamera:
    position: np.ndarray
    yaw_degrees: float = 0.0
    pitch_degrees: float = 0.0
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))

    def front(self) -> np.ndarray:
        yaw = math.radians(float(self.yaw_degrees))
        pitch = math.radians(float(np.clip(self.pitch_degrees, -89.0, 89.0)))
        return np.asarray(
            [
                math.sin(yaw) * math.cos(pitch),
                math.sin(pitch),
                math.cos(yaw) * math.cos(pitch),
            ],
            dtype=np.float32,
        )

    def right(self) -> np.ndarray:
        yaw = math.radians(float(self.yaw_degrees))
        return np.asarray([math.cos(yaw), 0.0, -math.sin(yaw)], dtype=np.float32)

    def up(self) -> np.ndarray:
        return np.asarray([0.0, 1.0, 0.0], dtype=np.float32)

    def target(self, distance: float = 1.0) -> np.ndarray:
        return self.position + self.front() * float(distance)

    def w2c(self) -> np.ndarray:
        return _look_at_w2c(self.position, self.target())

    def integrate(
        self,
        *,
        forward_axis: float,
        strafe_axis: float,
        vertical_axis: float,
        yaw_delta_degrees: float,
        pitch_delta_degrees: float,
        dt: float,
        speed: float,
        damping_tau: float,
    ) -> None:
        self.yaw_degrees = (self.yaw_degrees + float(yaw_delta_degrees) + 180.0) % 360.0 - 180.0
        self.pitch_degrees = float(np.clip(self.pitch_degrees + float(pitch_delta_degrees), -89.0, 89.0))

        forward = self.front()
        up = self.up()
        forward = forward - up * float(np.dot(forward, up))
        fn = float(np.linalg.norm(forward))
        if fn > 1e-6:
            forward /= fn
        desired = (
            forward * float(forward_axis)
            + self.right() * float(strafe_axis)
            + up * float(vertical_axis)
        )
        dn = float(np.linalg.norm(desired))
        if dn > 1e-6:
            desired = desired / dn * float(speed)
        else:
            desired[:] = 0.0

        tau = max(float(damping_tau), 1e-4)
        alpha = min(max(float(dt) / tau, 0.0), 1.0)
        self.velocity += (desired.astype(np.float32) - self.velocity) * alpha
        self.position = (self.position + self.velocity * float(dt)).astype(np.float32)


class FreeRoamRecorder:
    def __init__(self, image_hw: tuple[int, int], fov_degrees: float, record_hz: float) -> None:
        self.image_hw = image_hw
        self.K = _intrinsics_from_fov(image_hw, fov_degrees)
        self.record_period = 1.0 / max(float(record_hz), 1e-3)
        self._last_record_time = -1e9
        self.w2c: list[np.ndarray] = []
        self.Ks: list[np.ndarray] = []
        self.positions: list[np.ndarray] = []

    def maybe_record(self, camera: FreeRoamCamera, now: float, *, force: bool = False) -> None:
        if not force and now - self._last_record_time < self.record_period:
            return
        self._last_record_time = float(now)
        self.w2c.append(camera.w2c().astype(np.float32))
        self.Ks.append(self.K.copy())
        self.positions.append(camera.position.astype(np.float32).copy())

    def save(self, path: str | os.PathLike[str]) -> None:
        if not self.w2c:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            w2c=np.stack(self.w2c, axis=0).astype(np.float32),
            intrinsics=np.stack(self.Ks, axis=0).astype(np.float32),
            positions=np.stack(self.positions, axis=0).astype(np.float32),
            image_height=np.array(self.image_hw[0], dtype=np.int64),
            image_width=np.array(self.image_hw[1], dtype=np.int64),
        )


class PreviewPlayer:
    def __init__(self, manifest_path: str | None, fallback_fps: float) -> None:
        self.manifest_path = Path(manifest_path).resolve() if manifest_path else None
        self.fallback_fps = float(fallback_fps)
        self.manifest_mtime = 0.0
        self.frame_paths: list[Path] = []
        self.fps = float(fallback_fps)
        self.chunk_index = 0
        self.start_time = time.monotonic()
        self.current_index = -1
        self.sprite = None
        self.status = "waiting for generated preview frames"

    def _load_manifest_if_changed(self) -> None:
        if self.manifest_path is None:
            self.status = "preview disabled"
            return
        if not self.manifest_path.exists():
            self.status = f"waiting for {self.manifest_path}"
            return
        try:
            mtime = self.manifest_path.stat().st_mtime
            if mtime <= self.manifest_mtime:
                return
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self.status = f"preview manifest unreadable: {exc}"
            return

        base = self.manifest_path.parent
        frames = data.get("frames", [])
        if not isinstance(frames, list) or not frames:
            latest = data.get("latest_frame")
            frames = [latest] if isinstance(latest, str) else []
        paths = [(base / item).resolve() for item in frames if isinstance(item, str)]
        paths = [path for path in paths if path.exists()]
        if not paths:
            self.status = "preview manifest has no readable frames"
            return

        self.frame_paths = paths
        self.fps = max(float(data.get("fps", self.fallback_fps)), 1e-3)
        self.chunk_index = int(data.get("chunk_index", self.chunk_index + 1))
        self.manifest_mtime = mtime
        self.start_time = time.monotonic()
        self.current_index = -1
        self.status = f"playing chunk {self.chunk_index:04d} ({len(paths)} frames)"

    def update(self) -> None:
        self._load_manifest_if_changed()
        if not self.frame_paths:
            return
        elapsed = max(time.monotonic() - self.start_time, 0.0)
        index = int(elapsed * self.fps) % len(self.frame_paths)
        if index == self.current_index:
            return
        try:
            import pyglet

            image = pyglet.image.load(str(self.frame_paths[index]))
            self.sprite = pyglet.sprite.Sprite(image)
            self.current_index = index
        except Exception as exc:  # Pyglet image codecs can raise backend-specific errors.
            self.status = f"preview frame unreadable: {exc}"

    def draw(self, width: int, height: int) -> None:
        if self.sprite is None:
            return
        image = self.sprite.image
        scale = min(float(width) / max(float(image.width), 1.0), float(height) / max(float(image.height), 1.0))
        self.sprite.scale = scale
        self.sprite.x = 0.5 * (float(width) - float(image.width) * scale)
        self.sprite.y = 0.5 * (float(height) - float(image.height) * scale)
        self.sprite.draw()


def _atomic_write_json(path: str | os.PathLike[str], payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _command_payload(
    camera: FreeRoamCamera,
    *,
    sequence: int,
    forward_axis: float,
    strafe_axis: float,
    vertical_axis: float,
    yaw_velocity_degrees: float,
    pitch_velocity_degrees: float,
    image_hw: tuple[int, int],
    fov_degrees: float,
    trail: list[dict],
) -> dict:
    return {
        "schema": "lyra_free_roam_command.v1",
        "mode": "free_roam",
        "sequence": int(sequence),
        "time": time.time(),
        "position": [float(v) for v in camera.position],
        "yaw_degrees": float(camera.yaw_degrees),
        "pitch_degrees": float(camera.pitch_degrees),
        "forward": float(forward_axis),
        "strafe": float(strafe_axis),
        "vertical": float(vertical_axis),
        "turn_degrees": float(yaw_velocity_degrees),
        "pitch": float(pitch_velocity_degrees),
        "image_height": int(image_hw[0]),
        "image_width": int(image_hw[1]),
        "fov_degrees": float(fov_degrees),
        "w2c": camera.w2c().astype(float).tolist(),
        "trail": trail,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GL free-roam input controller for Lyra2 wos_walk streaming.")
    parser.add_argument("--command_path", type=str, default="outputs/free_roam/free_roam_command.json",
                        help="JSON command file consumed by --walk_control_mode free_roam.")
    parser.add_argument("--trajectory_out", type=str, default="outputs/free_roam/free_roam_trajectory.npz",
                        help="Recorded Lyra trajectory written on exit. Use '' to disable.")
    parser.add_argument("--resolution", type=_parse_hw, default=(256, 448), help="H,W used for recorded intrinsics.")
    parser.add_argument("--fov_degrees", type=float, default=60.0)
    parser.add_argument("--start_position", type=_parse_vec3, default=np.zeros(3, dtype=np.float32),
                        help="Initial x,y,z camera position in Lyra world units.")
    parser.add_argument("--start_yaw_degrees", type=float, default=0.0)
    parser.add_argument("--start_pitch_degrees", type=float, default=0.0)
    parser.add_argument("--speed", type=float, default=0.20, help="Base camera speed in Lyra world units/sec.")
    parser.add_argument("--fast_multiplier", type=float, default=3.0)
    parser.add_argument("--slow_multiplier", type=float, default=0.25)
    parser.add_argument("--turn_degrees_per_sec", type=float, default=75.0)
    parser.add_argument("--mouse_sensitivity", type=float, default=0.15, help="Degrees per mouse pixel.")
    parser.add_argument("--damping_tau", type=float, default=0.10)
    parser.add_argument("--write_hz", type=float, default=20.0)
    parser.add_argument("--record_hz", type=float, default=16.0)
    parser.add_argument("--trail_seconds", type=float, default=120.0,
                        help="Seconds of recent input trajectory kept in the command JSON.")
    parser.add_argument("--preview_manifest", type=str, default=None,
                        help="Lyra preview manifest to display in this GL window.")
    parser.add_argument("--preview_fps", type=float, default=16.0,
                        help="Fallback preview FPS if the manifest has no fps field.")
    parser.add_argument("--window_width", type=int, default=960)
    parser.add_argument("--window_height", type=int, default=540)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        import pyglet
        from pyglet.window import key, mouse
    except ImportError as exc:
        raise SystemExit("pyglet is required for the GL controller. Install with: uv pip install pyglet==2.1.14") from exc

    camera = FreeRoamCamera(
        position=args.start_position.astype(np.float32).copy(),
        yaw_degrees=float(args.start_yaw_degrees),
        pitch_degrees=float(args.start_pitch_degrees),
    )
    recorder = FreeRoamRecorder(args.resolution, args.fov_degrees, args.record_hz)
    recorder.maybe_record(camera, time.monotonic(), force=True)
    preview = PreviewPlayer(args.preview_manifest, args.preview_fps)

    window = pyglet.window.Window(
        width=int(args.window_width),
        height=int(args.window_height),
        caption="Lyra2 free-roam controller",
        resizable=True,
    )
    keys = key.KeyStateHandler()
    window.push_handlers(keys)

    state = {
        "sequence": 0,
        "last_write": -1e9,
        "last_forward": 0.0,
        "last_strafe": 0.0,
        "last_vertical": 0.0,
        "last_yaw_velocity": 0.0,
        "last_pitch_velocity": 0.0,
        "trail": [],
    }

    hud = pyglet.text.Label("", x=16, y=window.height - 24, anchor_x="left", anchor_y="top", multiline=True, width=760)

    def write_command(force: bool = False) -> None:
        now = time.monotonic()
        period = 1.0 / max(float(args.write_hz), 1e-3)
        if not force and now - state["last_write"] < period:
            return
        state["last_write"] = now
        state["sequence"] += 1
        sample = {
            "sequence": int(state["sequence"]),
            "monotonic_time": float(now),
            "wall_time": time.time(),
            "position": [float(v) for v in camera.position],
            "yaw_degrees": float(camera.yaw_degrees),
            "pitch_degrees": float(camera.pitch_degrees),
            "forward": float(state["last_forward"]),
            "strafe": float(state["last_strafe"]),
            "vertical": float(state["last_vertical"]),
        }
        state["trail"].append(sample)
        cutoff = now - max(float(args.trail_seconds), 1.0)
        state["trail"] = [item for item in state["trail"] if float(item.get("monotonic_time", now)) >= cutoff]
        _atomic_write_json(
            args.command_path,
            _command_payload(
                camera,
                sequence=state["sequence"],
                forward_axis=state["last_forward"],
                strafe_axis=state["last_strafe"],
                vertical_axis=state["last_vertical"],
                yaw_velocity_degrees=state["last_yaw_velocity"],
                pitch_velocity_degrees=state["last_pitch_velocity"],
                image_hw=args.resolution,
                fov_degrees=args.fov_degrees,
                trail=state["trail"],
            ),
        )

    def axes_from_keys(dt: float) -> tuple[float, float, float, float, float, float]:
        forward_axis = float(keys[key.W] or keys[key.UP]) - float(keys[key.S] or keys[key.DOWN])
        strafe_axis = float(keys[key.D]) - float(keys[key.A])
        vertical_axis = float(keys[key.E]) - float(keys[key.Q])
        yaw_axis = float(keys[key.RIGHT]) - float(keys[key.LEFT])
        pitch_axis = float(keys[key.R]) - float(keys[key.F])
        speed = float(args.speed)
        if keys[key.LSHIFT] or keys[key.RSHIFT]:
            speed *= float(args.fast_multiplier)
        if keys[key.LCTRL] or keys[key.RCTRL]:
            speed *= float(args.slow_multiplier)
        yaw_delta = yaw_axis * float(args.turn_degrees_per_sec) * float(dt)
        pitch_delta = pitch_axis * float(args.turn_degrees_per_sec) * float(dt)
        return forward_axis, strafe_axis, vertical_axis, yaw_delta, pitch_delta, speed

    def update(dt: float) -> None:
        forward_axis, strafe_axis, vertical_axis, yaw_delta, pitch_delta, speed = axes_from_keys(dt)
        camera.integrate(
            forward_axis=forward_axis,
            strafe_axis=strafe_axis,
            vertical_axis=vertical_axis,
            yaw_delta_degrees=yaw_delta,
            pitch_delta_degrees=pitch_delta,
            dt=dt,
            speed=speed,
            damping_tau=args.damping_tau,
        )
        state["last_forward"] = forward_axis
        state["last_strafe"] = strafe_axis
        state["last_vertical"] = vertical_axis
        state["last_yaw_velocity"] = yaw_delta / max(float(dt), 1e-6)
        state["last_pitch_velocity"] = pitch_delta / max(float(dt), 1e-6)
        recorder.maybe_record(camera, time.monotonic())
        write_command()
        preview.update()

    @window.event
    def on_mouse_drag(x, y, dx, dy, buttons, modifiers):  # noqa: ARG001
        if buttons & mouse.LEFT:
            camera.yaw_degrees = (camera.yaw_degrees - dx * float(args.mouse_sensitivity) + 180.0) % 360.0 - 180.0
            camera.pitch_degrees = float(np.clip(camera.pitch_degrees + dy * float(args.mouse_sensitivity), -89.0, 89.0))
            state["last_yaw_velocity"] = -dx * float(args.mouse_sensitivity)
            state["last_pitch_velocity"] = dy * float(args.mouse_sensitivity)
            write_command(force=True)

    @window.event
    def on_key_press(symbol, modifiers):  # noqa: ARG001
        if symbol == key.ESCAPE:
            window.close()
        elif symbol == key.SPACE:
            camera.velocity[:] = 0.0
            write_command(force=True)
        elif symbol == key.T:
            if args.trajectory_out:
                recorder.save(args.trajectory_out)
        elif symbol == key.HOME:
            camera.position[:] = args.start_position
            camera.yaw_degrees = float(args.start_yaw_degrees)
            camera.pitch_degrees = float(args.start_pitch_degrees)
            camera.velocity[:] = 0.0
            write_command(force=True)

    @window.event
    def on_draw():
        window.clear()
        preview.draw(window.width, window.height)
        p = camera.position
        hud.text = (
            "W/S forward/back  A/D strafe  Q/E down/up  Arrow left/right yaw  R/F pitch  "
            "Left-drag look  Shift fast  Ctrl slow  Space stop  T save  Home reset  Esc quit\n"
            f"command: {Path(args.command_path).resolve()}\n"
            f"preview: {preview.status}\n"
            f"pos=({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})  "
            f"yaw={camera.yaw_degrees:.1f}  pitch={camera.pitch_degrees:.1f}  "
            f"seq={state['sequence']}"
        )
        hud.y = window.height - 18
        hud.draw()

    @window.event
    def on_close():
        write_command(force=True)
        if args.trajectory_out:
            recorder.save(args.trajectory_out)

    write_command(force=True)
    pyglet.clock.schedule_interval(update, 1.0 / 60.0)
    pyglet.app.run()


if __name__ == "__main__":
    main()
