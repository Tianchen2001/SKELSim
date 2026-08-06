#!/usr/bin/env python3
"""Render a MuJoCo joint-monkey video for the generated SKEL asset.

Each of the 43 internal hinges is moved alone while all other coordinates
remain zero.  This is the fastest way to identify axis, sign, and coordinate
order errors before replaying a full walking sequence.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seconds-per-joint", type=float, default=0.7)
    parser.add_argument("--amplitude", type=float, default=0.35)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera-distance", type=float, default=3.0)
    return parser.parse_args()


def joint_address(model: mujoco.MjModel, name: str) -> int:
    joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, name
    )
    if joint_id < 0:
        raise KeyError(name)
    return int(model.jnt_qposadr[joint_id])


def main() -> None:
    args = parse_args()
    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    mapping = json.loads(
        Path(args.mapping).read_text(encoding="utf-8")
    )

    model.vis.global_.offwidth = max(
        int(model.vis.global_.offwidth), args.width
    )
    model.vis.global_.offheight = max(
        int(model.vis.global_.offheight), args.height
    )
    renderer = mujoco.Renderer(
        model, height=args.height, width=args.width
    )
    writer = imageio.get_writer(
        args.out,
        fps=args.fps,
        codec="libx264",
        quality=8,
        macro_block_size=None,
    )

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.asarray([0.0, 0.0, 0.9])
    camera.distance = args.camera_distance
    camera.azimuth = 135.0
    camera.elevation = -8.0

    frames_per_joint = max(
        1, int(round(args.seconds_per_joint * args.fps))
    )

    try:
        for entry in mapping["internal_coordinates"]:
            mujoco.mj_resetData(model, data)

            # Place the native Y-up asset upright without applying any
            # ForceBody motion.
            data.qpos[0:3] = np.asarray([0.0, 0.0, 0.95])
            half_sqrt = np.sqrt(0.5)
            data.qpos[3:7] = np.asarray(
                [half_sqrt, half_sqrt, 0.0, 0.0]
            )

            address = joint_address(
                model, entry["mjcf_joint_name"]
            )
            for frame_index in range(frames_per_joint):
                phase = 2.0 * np.pi * frame_index / frames_per_joint
                data.qpos[address] = args.amplitude * np.sin(phase)
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera)
                writer.append_data(renderer.render())

            print(
                f"[joint] pose[{entry['pose_index']:02d}] "
                f"{entry['pose_name']} -> "
                f"{entry['mjcf_joint_name']}"
            )
    finally:
        writer.close()
        renderer.close()

    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
