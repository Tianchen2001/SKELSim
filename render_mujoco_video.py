#!/usr/bin/env python3
"""Render a MuJoCo MJCF model to MP4 without opening a viewer."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", required=True)
    parser.add_argument("--output", default="mujoco_video.mp4")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument(
        "--mode", choices=["static", "passive"], default="passive",
        help="static: keep qpos fixed and orbit camera; passive: run mj_step.",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--keyframe", type=int, default=None)
    parser.add_argument("--azimuth", type=float, default=135.0)
    parser.add_argument("--elevation", type=float, default=-15.0)
    parser.add_argument("--distance-scale", type=float, default=2.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    xml_path = Path(args.xml).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not xml_path.is_file():
        raise FileNotFoundError(f"MJCF not found: {xml_path}")
    if args.seconds <= 0 or args.fps <= 0:
        raise ValueError("--seconds and --fps must be positive.")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), args.width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), args.height)

    if args.keyframe is not None:
        if not 0 <= args.keyframe < model.nkey:
            raise ValueError(
                f"Keyframe {args.keyframe} invalid; model has {model.nkey}."
            )
        mujoco.mj_resetDataKeyframe(model, data, args.keyframe)
    else:
        mujoco.mj_resetData(model, data)

    mujoco.mj_forward(model, data)

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = model.stat.center
    camera.distance = max(args.distance_scale * model.stat.extent, 1.0)
    camera.azimuth = args.azimuth
    camera.elevation = args.elevation

    renderer = mujoco.Renderer(
        model,
        height=args.height,
        width=args.width,
    )

    frame_interval = 1.0 / args.fps
    next_frame_time = 0.0
    frame_count = 0

    writer = imageio.get_writer(
        str(output_path),
        fps=args.fps,
        codec="libx264",
        quality=8,
        macro_block_size=None,
    )

    try:
        if args.mode == "static":
            total_frames = int(np.ceil(args.seconds * args.fps))
            initial_azimuth = camera.azimuth
            for frame_index in range(total_frames):
                camera.azimuth = initial_azimuth + 360.0 * frame_index / max(total_frames, 1)
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera)
                writer.append_data(renderer.render())
                frame_count += 1
        else:
            while data.time < args.seconds:
                mujoco.mj_step(model, data)
                if data.time + 1e-12 >= next_frame_time:
                    renderer.update_scene(data, camera=camera)
                    writer.append_data(renderer.render())
                    frame_count += 1
                    next_frame_time = frame_count * frame_interval
    finally:
        writer.close()
        renderer.close()

    print(f"MuJoCo version: {mujoco.__version__}")
    print(f"Backend: {os.environ.get('MUJOCO_GL')}")
    print(f"Mode: {args.mode}")
    print(f"Rendered frames: {frame_count}")
    print(f"Saved video: {output_path}")


if __name__ == "__main__":
    main()
