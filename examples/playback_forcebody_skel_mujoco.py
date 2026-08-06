#!/usr/bin/env python3
"""Kinematically replay a ForceBody SKEL sequence in MuJoCo and render MP4.

This script does not call ``mj_step``.  Every output frame is produced by:
    ForceBody poses/trans -> MuJoCo qpos -> mujoco.mj_forward()

This isolates SKEL-to-MJCF kinematics from mass, contact, actuators, numerical
integration, and controller behavior.  It also compares MuJoCo body origins
against the 24 anatomical joint positions returned by the official SKEL
forward pass.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

# The OpenGL backend must be selected before importing mujoco.
os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from skel.skel_model import SKEL
from smpl_sim.smpllib.skel_joint_spec import SKEL_BODY_NAMES


# Rotate a Y-up SKEL model into a conventional Z-up MuJoCo world:
# (x, y, z)_skel -> (x, -z, y)_mujoco.
R_YUP_TO_ZUP = np.asarray(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument("--xml", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument(
        "--skel-model-path",
        required=True,
        help="Directory containing skel_male.pkl and skel_female.pkl.",
    )
    parser.add_argument("--out", required=True, help="Output MP4 path.")
    parser.add_argument(
        "--metrics-csv",
        default=None,
        help="Optional per-frame FK error CSV.",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--azimuth", type=float, default=135.0)
    parser.add_argument("--elevation", type=float, default=-8.0)
    parser.add_argument("--camera-distance", type=float, default=3.0)
    parser.add_argument(
        "--disable-render",
        action="store_true",
        help="Compute FK metrics without creating a video.",
    )
    return parser.parse_args()


def _normalize_gender(value: object) -> str:
    text = str(value).strip().lower()
    return "female" if "female" in text or text == "f" else "male"


def _matrix_to_wxyz(matrix: np.ndarray) -> np.ndarray:
    xyzw = Rotation.from_matrix(matrix).as_quat()
    return np.asarray([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])


def _joint_qpos_address(model: mujoco.MjModel, joint_name: str) -> int:
    joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
    )
    if joint_id < 0:
        raise KeyError(f"Joint not found in MJCF: {joint_name}")
    return int(model.jnt_qposadr[joint_id])


def _body_id(model: mujoco.MjModel, body_name: str) -> int:
    body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, body_name
    )
    if body_id < 0:
        raise KeyError(f"Body not found in MJCF: {body_name}")
    return int(body_id)


def main() -> None:
    args = parse_args()
    npz_path = Path(args.npz).expanduser().resolve()
    xml_path = Path(args.xml).expanduser().resolve()
    mapping_path = Path(args.mapping).expanduser().resolve()
    output_path = Path(args.out).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sample = np.load(npz_path, allow_pickle=True)
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))

    poses_np = np.asarray(sample["poses"], dtype=np.float32)
    trans_np = np.asarray(sample["trans"], dtype=np.float32)
    betas_np = np.asarray(sample["betas"], dtype=np.float32).reshape(1, 10)
    source_fps = float(sample["fps"])
    frame_step = max(1, int(round(source_fps / args.fps)))
    frame_indices = list(range(0, len(poses_np), frame_step))
    if args.max_frames > 0:
        frame_indices = frame_indices[: args.max_frames]

    gender = _normalize_gender(
        sample["sex"] if "sex" in sample.files else mapping["gender"]
    )
    skel = SKEL(gender=gender, model_path=args.skel_model_path)

    selected_poses = torch.from_numpy(poses_np[frame_indices])
    selected_trans = torch.from_numpy(trans_np[frame_indices])
    selected_betas = torch.from_numpy(betas_np).expand(
        len(frame_indices), -1
    )

    with torch.no_grad():
        skel_output = skel.forward(
            poses=selected_poses,
            betas=selected_betas,
            trans=selected_trans,
            skelmesh=False,
            pose_dep_bs=True,
        )
        zero_output = skel.forward(
            poses=torch.zeros((1, 46), dtype=torch.float32),
            betas=torch.from_numpy(betas_np),
            trans=torch.zeros((1, 3), dtype=torch.float32),
            skelmesh=False,
            pose_dep_bs=True,
        )

    skel_joints = skel_output.joints.detach().cpu().numpy()
    skel_orientations = skel_output.joints_ori.detach().cpu().numpy()
    zero_root_position = (
        zero_output.joints_tpose[0, 0].detach().cpu().numpy()
    )
    zero_root_orientation = (
        zero_output.joints_ori[0, 0].detach().cpu().numpy()
    )

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    if model.nq != 7 + 43:
        raise RuntimeError(
            f"Expected nq=50 (free root + 43 hinges), got nq={model.nq}."
        )

    root_qpos_address = _joint_qpos_address(
        model, mapping["root"]["joint_name"]
    )
    if root_qpos_address != 0:
        raise RuntimeError(
            f"Expected the free root at qpos address 0, got "
            f"{root_qpos_address}."
        )

    coordinate_entries = mapping["internal_coordinates"]
    joint_addresses = {
        entry["mjcf_joint_name"]: _joint_qpos_address(
            model, entry["mjcf_joint_name"]
        )
        for entry in coordinate_entries
    }
    body_ids = np.asarray(
        [_body_id(model, name) for name in SKEL_BODY_NAMES],
        dtype=np.int32,
    )

    renderer = None
    writer = None
    if not args.disable_render:
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
            str(output_path),
            fps=args.fps,
            codec="libx264",
            quality=8,
            macro_block_size=None,
        )

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth = args.azimuth
    camera.elevation = args.elevation
    camera.distance = args.camera_distance

    metrics: list[dict[str, float | int]] = []

    try:
        for output_index, source_index in enumerate(frame_indices):
            mujoco.mj_resetData(model, data)

            # Root position: the generated XML is pelvis-centered, so remove
            # the subject's zero-pose pelvis offset before changing world basis.
            root_position_skel = (
                skel_joints[output_index, 0] - zero_root_position
            )
            data.qpos[0:3] = R_YUP_TO_ZUP @ root_position_skel

            # Root orientation: relative SKEL orientation followed by the
            # one-time Y-up to Z-up world alignment.
            relative_root_R = (
                skel_orientations[output_index, 0]
                @ zero_root_orientation.T
            )
            root_R_mujoco = R_YUP_TO_ZUP @ relative_root_R
            data.qpos[3:7] = _matrix_to_wxyz(root_R_mujoco)

            for entry in coordinate_entries:
                pose_index = int(entry["pose_index"])
                address = joint_addresses[entry["mjcf_joint_name"]]
                data.qpos[address] = float(
                    poses_np[source_index, pose_index]
                )

            mujoco.mj_forward(model, data)

            target_positions = (
                (skel_joints[output_index] - zero_root_position)
                @ R_YUP_TO_ZUP.T
            )
            mujoco_positions = data.xpos[body_ids].copy()
            per_body_error = np.linalg.norm(
                mujoco_positions - target_positions, axis=1
            )

            metrics.append(
                {
                    "output_frame": output_index,
                    "source_frame": source_index,
                    "time_s": source_index / source_fps,
                    "mean_joint_error_m": float(
                        np.mean(per_body_error)
                    ),
                    "max_joint_error_m": float(
                        np.max(per_body_error)
                    ),
                    "pelvis_error_m": float(per_body_error[0]),
                    "right_foot_error_m": float(per_body_error[4]),
                    "left_foot_error_m": float(per_body_error[9]),
                    "right_scapula_error_m": float(per_body_error[14]),
                    "left_scapula_error_m": float(per_body_error[19]),
                }
            )

            if renderer is not None and writer is not None:
                root_world = data.xpos[body_ids[0]]
                camera.lookat[:] = root_world + np.asarray(
                    [0.0, 0.0, 0.75]
                )
                renderer.update_scene(data, camera=camera)
                writer.append_data(renderer.render())

            if output_index % 30 == 0:
                print(
                    f"[frame {output_index:04d}] "
                    f"mean={metrics[-1]['mean_joint_error_m']:.4f} m, "
                    f"max={metrics[-1]['max_joint_error_m']:.4f} m"
                )
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()

    mean_errors = np.asarray(
        [row["mean_joint_error_m"] for row in metrics]
    )
    max_errors = np.asarray(
        [row["max_joint_error_m"] for row in metrics]
    )
    print(f"[summary] frames: {len(metrics)}")
    print(
        f"[summary] mean FK error over sequence: "
        f"{float(np.mean(mean_errors)):.6f} m"
    )
    print(
        f"[summary] worst per-frame max error: "
        f"{float(np.max(max_errors)):.6f} m"
    )
    if not args.disable_render:
        print(f"[saved] video: {output_path}")

    csv_path = (
        Path(args.metrics_csv).expanduser().resolve()
        if args.metrics_csv
        else output_path.with_suffix(".fk_metrics.csv")
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer_csv = csv.DictWriter(
            handle, fieldnames=list(metrics[0].keys())
        )
        writer_csv.writeheader()
        writer_csv.writerows(metrics)
    print(f"[saved] FK metrics: {csv_path}")


if __name__ == "__main__":
    main()
