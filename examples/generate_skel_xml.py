#!/usr/bin/env python3
"""Generate a subject-specific SKEL MJCF from a ForceBody sample.

The XML contains the body tree, mesh geometries, a free root, 43 internal
hinge joints, and motor definitions.  It does *not* contain the walking
trajectory.  Use ``playback_forcebody_skel_mujoco.py`` to stream the sample's
``poses`` and ``trans`` into MuJoCo qpos.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from smpl_sim.smpllib.skel_local_robot import SKEL_Robot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True, help="ForceBody NPZ sample.")
    parser.add_argument(
        "--skel-model-path",
        required=True,
        help="Directory containing skel_male.pkl and skel_female.pkl.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--flatfoot",
        action="store_true",
        help="Flatten the lowest 1 cm of the zero-pose skin mesh.",
    )
    parser.add_argument(
        "--forward-joint-order",
        action="store_true",
        help=(
            "Write multi-DOF coordinates in pose order instead of the default "
            "reverse order. Use this only as an FK debugging alternative."
        ),
    )
    parser.add_argument(
        "--normalize-total-mass",
        action="store_true",
        help="Scale all mesh densities to match NPZ mass_kg.",
    )
    parser.add_argument("--verbose-geometry", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    robot = SKEL_Robot.from_forcebody_npz(
        args.npz,
        skel_model_path=args.skel_model_path,
        output_dir=out_dir,
        reverse_multidof_order=not args.forward_joint_order,
    )
    robot.build(
        flatfoot=args.flatfoot,
        verbose_geometry=args.verbose_geometry,
    )

    xml_path = robot.write_xml(out_dir / "skel_humanoid.xml")
    mapping_path = robot.write_mapping(
        out_dir / "skel_humanoid.mapping.json"
    )

    print(f"[saved] XML:     {xml_path}")
    print(f"[saved] mapping: {mapping_path}")
    print(f"[saved] meshes:  {robot.geom_dir}")

    if args.normalize_total_mass:
        compiled_mass = robot.normalize_xml_mass(xml_path)
        print(f"[mass] compiled total after scaling: {compiled_mass:.6f} kg")

    print(
        "[next] Run playback_forcebody_skel_mujoco.py with the same NPZ, "
        "XML, mapping JSON, and SKEL model directory."
    )


if __name__ == "__main__":
    main()
