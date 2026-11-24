#!/usr/bin/env python3
"""Locate where the SKEL -> MuJoCo forward-kinematics error comes from.

Same replay as playback_forcebody_skel_mujoco.py (no rendering), but the error is reported per body:

  zero     |SKEL joints at q = 0  -  SKEL joints_tpose|. The XML body offsets come from joints_tpose,
           so this part of the error is already there before any joint moves.
  global   error of the body origin in the world (accumulates along the chain).
  local    error of the vector parent origin -> body origin. A wrong axis or rotation order at the
           PARENT joint, or a missing translation at this joint, shows up here and only here.
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from skel.skel_model import SKEL

from smpl_sim.smpllib.skel_joint_spec import SKEL_BODY_NAMES, SKEL_PARENT_INDICES

R_YUP_TO_ZUP = np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True)
    p.add_argument("--xml", required=True)
    p.add_argument("--mapping", required=True)
    p.add_argument("--skel-model-path", required=True)
    p.add_argument("--step", type=int, default=10, help="use every n-th frame")
    args = p.parse_args()

    sample = np.load(args.npz, allow_pickle=True)
    mapping = json.loads(open(args.mapping, encoding="utf-8").read())
    poses = np.asarray(sample["poses"], dtype=np.float32)[:: args.step]
    trans = np.asarray(sample["trans"], dtype=np.float32)[:: args.step]
    betas = np.asarray(sample["betas"], dtype=np.float32).reshape(1, 10)
    gender = "female" if "female" in str(sample["skel_gender"] if "skel_gender" in sample.files else mapping["gender"]).lower() else "male"

    skel = SKEL(gender=gender, model_path=args.skel_model_path)
    with torch.no_grad():
        out = skel.forward(poses=torch.from_numpy(poses), betas=torch.from_numpy(betas).expand(len(poses), -1),
                           trans=torch.from_numpy(trans), skelmesh=False, pose_dep_bs=True)
        zero = skel.forward(poses=torch.zeros((1, 46)), betas=torch.from_numpy(betas), trans=torch.zeros((1, 3)),
                            skelmesh=False, pose_dep_bs=True)
    joints = out.joints.numpy().astype(np.float64)              # (T, 24, 3)
    ori = out.joints_ori.numpy().astype(np.float64)             # (T, 24, 3, 3)
    zero_joints = zero.joints[0].numpy().astype(np.float64)
    zero_tpose = zero.joints_tpose[0].numpy().astype(np.float64)
    zero_root, zero_root_R = zero_tpose[0], zero.joints_ori[0, 0].numpy().astype(np.float64)

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    body = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in SKEL_BODY_NAMES])
    adr = {e["mjcf_joint_name"]: int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, e["mjcf_joint_name"])])
           for e in mapping["internal_coordinates"]}
    parent = np.array([max(i, 0) for i in SKEL_PARENT_INDICES])

    glob, loc = [], []
    for t in range(len(poses)):
        mujoco.mj_resetData(model, data)
        data.qpos[0:3] = R_YUP_TO_ZUP @ (joints[t, 0] - zero_root)
        xyzw = Rotation.from_matrix(R_YUP_TO_ZUP @ (ori[t, 0] @ zero_root_R.T)).as_quat()
        data.qpos[3:7] = xyzw[[3, 0, 1, 2]]
        for e in mapping["internal_coordinates"]:
            data.qpos[adr[e["mjcf_joint_name"]]] = float(poses[t, int(e["pose_index"])])
        mujoco.mj_forward(model, data)
        mj = data.xpos[body].copy()
        target = (joints[t] - zero_root) @ R_YUP_TO_ZUP.T
        glob.append(np.linalg.norm(mj - target, axis=1))
        loc.append(np.linalg.norm((mj - mj[parent]) - (target - target[parent]), axis=1))
    glob, loc = np.array(glob) * 1000, np.array(loc) * 1000
    zero_err = np.linalg.norm(zero_joints - zero_tpose, axis=1) * 1000
    zero_loc = np.linalg.norm((zero_joints - zero_joints[parent]) - (zero_tpose - zero_tpose[parent]), axis=1) * 1000

    print(f"frames {len(poses)}, gender {gender}; all numbers in mm")
    print(f"{'body':12s} {'parent':12s} {'zero':>7s} {'zero loc':>9s} | {'global mean':>11s} {'max':>6s} | {'local mean':>10s} {'max':>6s} {'std':>6s}")
    for i, name in enumerate(SKEL_BODY_NAMES):
        par = SKEL_BODY_NAMES[SKEL_PARENT_INDICES[i]] if SKEL_PARENT_INDICES[i] >= 0 else "-"
        print(f"{name:12s} {par:12s} {zero_err[i]:7.1f} {zero_loc[i]:9.1f} | {glob[:, i].mean():11.1f} {glob[:, i].max():6.1f} | "
              f"{loc[:, i].mean():10.1f} {loc[:, i].max():6.1f} {loc[:, i].std():6.1f}")
    print(f"mean over bodies: global {glob.mean():.1f}, local {loc.mean():.1f}, zero {zero_err.mean():.1f}")


if __name__ == "__main__":
    main()
