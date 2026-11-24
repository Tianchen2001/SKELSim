#!/usr/bin/env python3
"""Convert one ForceBody trial into a clip for a SKEL dynamics model (build_skel_dynamics_model.py).

The clip holds the model qpos per frame (free root + hinges, model frame x forward / y left / z up), the
measured ground reaction per foot in the same frame, and the ForceBody torque labels with their masks.
With --rescale the trial is fitted to a model of another size: positions (root, centre of pressure) are
multiplied by the leg-length ratio, forces by the mass ratio, moments by both.  Joint angles and time are kept.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from skel.skel_model import SKEL

# SKEL pose name -> ForceBody (OpenSim) torque column, for the coordinates that mean the same thing.
LABELS = {
    "hip_flexion_{s}": "hip_{s}_0", "hip_adduction_{s}": "hip_{s}_1", "hip_rotation_{s}": "hip_{s}_2",
    "knee_angle_{s}": "walker_knee_{s}_0", "ankle_angle_{s}": "ankle_{s}_0",
    "subtalar_angle_{s}": "subtalar_{s}_0", "mtp_angle_{s}": "mtp_{s}_0",
    "elbow_flexion_{s}": "elbow_{s}_0", "pro_sup_{s}": "radioulnar_{s}_0",
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz", required=True, type=Path)
    p.add_argument("--xml", required=True, type=Path, help="dynamics model; its .json must sit next to it")
    p.add_argument("--skel-model-path", required=True)
    p.add_argument("--pose-source", choices=["kin", "dyn"], default="dyn",
                   help="dyn: poses_dyn, fitted to the AddB dynamics pass q_gt on which the torque labels are computed "
                        "(tau_sc_kinematics = q_gt); kin: poses, fitted to the kinematics pass")
    p.add_argument("--rescale", action="store_true", help="scale lengths by leg length and forces by mass to the model")
    p.add_argument("--belt", choices=["stored", "none"], default="stored",
                   help="treadmill trials: stored = move the body (and the centres of pressure) against the belt with the "
                        "belt velocity stored in the trial, so that the stance foot rests on a fixed floor; none = walk in place")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    desc = json.loads(args.xml.with_suffix(".json").read_text())
    R = np.array(desc["old_to_new_rotation"])
    d = np.load(args.npz, allow_pickle=True)
    suffix = "" if args.pose_source == "kin" else "_dyn"
    poses, trans = np.asarray(d["poses" + suffix], np.float32), np.asarray(d["trans" + suffix], np.float32)
    betas = np.asarray(d["betas"], np.float32).reshape(1, 10)
    gender = "female" if "female" in str(d["skel_gender"]).lower() else "male"
    fps = float(d["fps_true_grid"]) if "fps_true_grid" in d.files else float(d["fps"])
    T = len(poses)
    same_body = gender == desc["gender"] and np.abs(betas[0] - np.array(desc["betas"])).max() <= 1e-4
    if not same_body and not args.rescale:
        print(f"[warning] the model is {desc['gender']} with other betas than this trial ({gender}); the motion is not rescaled (see --rescale)")

    skel = SKEL(gender=gender, model_path=args.skel_model_path)
    root_pos, root_R = [], []
    with torch.no_grad():
        zero = skel.forward(poses=torch.zeros((1, 46)), betas=torch.from_numpy(betas), trans=torch.zeros((1, 3)), skelmesh=False, pose_dep_bs=True)
        zero_R = zero.joints_ori[0, 0].numpy().astype(np.float64)
        for a in range(0, T, 200):
            out = skel.forward(poses=torch.from_numpy(poses[a:a + 200]), betas=torch.from_numpy(betas).expand(len(poses[a:a + 200]), -1),
                               trans=torch.from_numpy(trans[a:a + 200]), skelmesh=False, pose_dep_bs=True)
            root_pos.append(out.joints[:, 0].numpy().astype(np.float64))
            root_R.append(out.joints_ori[:, 0].numpy().astype(np.float64))
    root_pos, root_R = np.concatenate(root_pos), np.concatenate(root_R)

    length_scale = force_scale = 1.0
    if args.rescale:
        leg = lambda p, side: float(np.linalg.norm(p[f"tibia_{side}"] - p[f"femur_{side}"]) + np.linalg.norm(p[f"talus_{side}"] - p[f"tibia_{side}"]))
        zj = zero.joints[0].numpy().astype(np.float64)
        subject = {n: zj[i] for i, n in ((1, "femur_r"), (2, "tibia_r"), (3, "talus_r"), (6, "femur_l"), (7, "tibia_l"), (8, "talus_l"))}
        model = mujoco.MjModel.from_xml_path(str(args.xml)); mdata = mujoco.MjData(model); mujoco.mj_forward(model, mdata)
        mpos = {n: mdata.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)].copy() for n in subject}
        leg_subject, leg_model = 0.5 * (leg(subject, "r") + leg(subject, "l")), 0.5 * (leg(mpos, "r") + leg(mpos, "l"))
        length_scale = leg_model / leg_subject
        force_scale = float(model.body_mass.sum()) / float(d["mass_kg_dynamics"])
        print(f"[rescale] leg length {leg_subject:.3f} m -> {leg_model:.3f} m (x {length_scale:.3f}); mass {float(d['mass_kg_dynamics']):.1f} kg -> {float(model.body_mass.sum()):.1f} kg (x {force_scale:.3f})")

    joints = desc["joints"]
    qpos = np.zeros((T, 7 + len(joints)))
    qpos[:, :3] = length_scale * (root_pos @ R.T)
    xyzw = Rotation.from_matrix(np.einsum("ij,tjk,lk->til", R, root_R @ zero_R.T, R)).as_quat()
    qpos[:, 3:7] = xyzw[:, [3, 0, 1, 2]]
    for k, j in enumerate(joints):
        qpos[:, 7 + k] = poses[:, j["pose_index"]]

    grf = force_scale * (np.asarray(d["grf_aligned"], np.float64)[:, :2] @ R.T)
    cop = length_scale * (np.asarray(d["cop_aligned_foot"], np.float64)[:, :2] @ R.T)

    # Treadmill: the recording is in the lab frame, where the subject walks in place on a moving belt.  In the
    # frame that moves with the belt the floor is at rest and the subject walks forward; with a constant belt
    # speed this changes no acceleration, so the inverse dynamics is the same in both frames.
    setting = str(np.asarray(d["setting"]).reshape(-1)[0]).lower() if "setting" in d.files else ""
    belt_velocity = np.zeros(3)
    if setting == "treadmill" and args.belt == "stored":
        belt_velocity = length_scale * (R @ np.array([float(d["belt_vx_mps"]), 0.0, float(d["belt_vz_mps"])]))
        model = mujoco.MjModel.from_xml_path(str(args.xml)); mdata = mujoco.MjData(model)
        feet = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("calcn_r", "calcn_l")]
        foot = np.zeros((T, 2, 2))
        for t in range(T):
            mdata.qpos[:] = qpos[t]; mujoco.mj_kinematics(model, mdata); foot[t] = mdata.xpos[feet, :2]
        foot_v = np.gradient(foot, 1.0 / fps, axis=0)
        stance = grf[:, :, 2] > 0.2 * float(grf[:, :, 2].sum(1).mean())
        slip = lambda v: float(np.median(np.linalg.norm(foot_v - v, axis=-1)[stance])) if stance.any() else float("nan")
        before, after = slip(0.0), slip(belt_velocity[:2])
        print(f"[belt] stored belt velocity {np.linalg.norm(belt_velocity):.3f} m/s (after length scaling); stance-foot speed "
              f"relative to the floor: {before:.3f} m/s in the lab frame, {after:.3f} m/s in the belt frame")
        if after > before:
            print("[belt] WARNING: the stored belt velocity increases the foot slip; left unapplied")
            belt_velocity = np.zeros(3)
        else:
            shift = -np.arange(T)[:, None] / fps * belt_velocity[None, :2]
            qpos[:, :2] += shift
            cop[:, :, :2] += shift[:, None, :]
            print(f"[belt] applied: the root travels {np.linalg.norm(qpos[-1, :2] - qpos[0, :2]):.2f} m in {T / fps:.1f} s")

    names = ['pelvis_tilt','pelvis_list','pelvis_rotation','hip_flexion_r','hip_adduction_r','hip_rotation_r','knee_angle_r','ankle_angle_r','subtalar_angle_r','mtp_angle_r','hip_flexion_l','hip_adduction_l','hip_rotation_l','knee_angle_l','ankle_angle_l','subtalar_angle_l','mtp_angle_l','lumbar_bending','lumbar_extension','lumbar_twist','thorax_bending','thorax_extension','thorax_twist','head_bending','head_extension','head_twist','scapula_abduction_r','scapula_elevation_r','scapula_upward_rot_r','shoulder_r_x','shoulder_r_y','shoulder_r_z','elbow_flexion_r','pro_sup_r','wrist_flexion_r','wrist_deviation_r','scapula_abduction_l','scapula_elevation_l','scapula_upward_rot_l','shoulder_l_x','shoulder_l_y','shoulder_l_z','elbow_flexion_l','pro_sup_l','wrist_flexion_l','wrist_deviation_l']
    for name, value in desc["locked_pose_values_rad"].items():
        dev = np.degrees(np.abs(poses[:, names.index(name)] - value).max())
        if dev > 1.0:
            print(f"[warning] {name} is locked at {np.degrees(value):.1f} deg in the model but moves up to {dev:.1f} deg away from it in this trial")

    dof_names = [str(n) for n in d["dof_names"]]
    q_name = "q_gt" if args.pose_source == "dyn" else "q_kin"
    q_kin = np.asarray(d[q_name], np.float64)
    pairs = []
    if "tau_sc_kinematics" in d.files and str(d["tau_sc_kinematics"]) != q_name:
        print(f"[warning] the torque labels of this trial were computed on {d['tau_sc_kinematics']}, the clip uses the fit to {q_name}")
    print(f"[labels] SKEL coordinate vs ForceBody coordinate ({q_name}): correlation, mean difference in deg")
    for side in "rl":
        for skel_name, fb_name in LABELS.items():
            s, f = skel_name.format(s=side), fb_name.format(s=side)
            a, b = poses[:, names.index(s)].astype(np.float64), q_kin[:, dof_names.index(f)]
            r = float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-6 and b.std() > 1e-6 else float("nan")
            sign = -1.0 if r < 0 else 1.0
            pairs.append({"skel": s, "forcebody": f, "column": dof_names.index(f), "sign": sign, "r_angle": r})
            print(f"         {s:20s} {f:18s} r {r:6.3f}   mean diff {np.degrees(np.mean(a - sign * b)):6.1f}   {'(constant in one of them)' if np.isnan(r) else ''}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out, qpos=qpos, fps=fps, joint_names=np.array([j["mjcf_joint_name"] for j in joints]), pose_names=np.array([j["pose_name"] for j in joints]),
        grf=grf, cop=cop, belt_velocity=belt_velocity, setting=setting,
        free_moment=force_scale * length_scale * (np.asarray(d["contact_torque_aligned"], np.float64)[:, :2] @ R.T),
        length_scale=length_scale, force_scale=force_scale,
        tau=np.asarray(d["tau_selfconsistent"], np.float64), sigma_tau=np.asarray(d["sigma_tau"], np.float64),
        tau_frame_ok=np.asarray(d["tau_frame_ok"], bool), mc_frame_ok=np.asarray(d["mc_frame_ok"], bool), grf_frame_ok=np.asarray(d["grf_frame_ok"], bool),
        cop_frame_ok=np.asarray(d["cop_frame_ok"], bool) if "cop_frame_ok" in d.files else np.ones(T, bool),
        mass_kg_dynamics=float(d["mass_kg_dynamics"]), label_pairs=json.dumps(pairs), source=str(args.npz), pose_source=args.pose_source)
    print(f"[saved] {args.out}: {T} frames at {fps:.2f} Hz, pelvis height {qpos[:, 2].min():.3f}-{qpos[:, 2].max():.3f} m, "
          f"lowest centre of pressure {np.asarray(d['cop_aligned_foot'])[:, :2, 1].min():.3f} m (floor should be near 0)")


if __name__ == "__main__":
    main()
