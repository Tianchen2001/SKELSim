#!/usr/bin/env python3
"""Check the collision shapes of a SKEL dynamics model against reference clips (kinematic replay).

A reference motion should not make the body shapes penetrate each other, and only the feet should touch the
floor.  For every clip the poses are put on the model and the contacts are listed by body pair: share of
frames in contact and deepest penetration.  Pairs that show up here are shapes that are too fat for the data.
"""
from __future__ import annotations

import argparse
import collections
import glob

import mujoco
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--xml", required=True)
    p.add_argument("--clips", required=True, nargs="+")
    p.add_argument("--step", type=int, default=2, help="use every n-th frame")
    p.add_argument("--depth", type=float, default=0.002, help="report penetrations deeper than this, m")
    args = p.parse_args()

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    files = sorted(f for pattern in args.clips for f in glob.glob(pattern))
    frames_total = 0
    pair_frames, pair_depth, pair_clips = collections.Counter(), collections.defaultdict(float), collections.defaultdict(set)
    feet = ("calcn_r", "calcn_l", "toes_r", "toes_l")
    foot_low = []
    for file in files:
        qpos = np.load(file, allow_pickle=True)["qpos"][:: args.step]
        if qpos.shape[1] != model.nq:
            continue
        for q in qpos:
            data.qpos[:] = q
            mujoco.mj_kinematics(model, data); mujoco.mj_collision(model, data)
            frames_total += 1
            seen = {}
            for c in data.contact[: data.ncon]:
                a, b = model.body(model.geom_bodyid[c.geom1]).name, model.body(model.geom_bodyid[c.geom2]).name
                if a == "world" or b == "world":
                    other = b if a == "world" else a
                    if other in feet:
                        foot_low.append(c.dist)
                        continue
                    key = ("floor", other)
                else:
                    key = tuple(sorted((a, b)))
                if c.dist < -args.depth:
                    seen[key] = min(seen.get(key, 0.0), c.dist)
            for key, dist in seen.items():
                pair_frames[key] += 1; pair_depth[key] = min(pair_depth[key], dist); pair_clips[key].add(file)
    print(f"[collision] {len(files)} clips, {frames_total} frames checked on {args.xml}")
    if foot_low:
        print(f"            feet against the floor: deepest {-1000 * min(foot_low):.0f} mm, 95th percentile {-1000 * np.percentile(foot_low, 5):.0f} mm below the surface")
    if not pair_frames:
        print(f"            no other penetration deeper than {1000 * args.depth:.0f} mm")
    else:
        print(f"            {'body pair':28s} {'frames':>8s} {'clips':>6s} {'deepest':>9s}")
        for key, n in pair_frames.most_common():
            print(f"            {key[0] + ' - ' + key[1]:28s} {100 * n / frames_total:7.2f}% {len(pair_clips[key]):6d} {-1000 * pair_depth[key]:7.0f} mm")


if __name__ == "__main__":
    main()
