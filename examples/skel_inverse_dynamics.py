#!/usr/bin/env python3
"""Inverse dynamics of a clip on a SKEL dynamics model, compared with the ForceBody torque labels.

For every frame the recorded pose, its velocity and acceleration (finite differences, low-passed) are put on
the model, the measured ground reaction is applied to the feet at the centre of pressure, and MuJoCo
returns the generalized forces that explain the motion: 6 on the free root (the residual, ideally 0) and
one torque per hinge.  Contacts, joint limits and actuators of the model are not used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from scipy.signal import butter, filtfilt


def lowpass(x: np.ndarray, cutoff: float, rate: float) -> np.ndarray:
    if cutoff <= 0 or len(x) < 30:
        return x
    b, a = butter(4, cutoff / (0.5 * rate))
    return filtfilt(b, a, x, axis=0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--xml", required=True, type=Path)
    p.add_argument("--clip", required=True, type=Path)
    p.add_argument("--out", type=Path, default=None, help="output prefix (default: next to the clip)")
    p.add_argument("--cutoff", type=float, default=10.0, help="low-pass for velocities, forces and labels, Hz; 0 = off")
    p.add_argument("--min-force", type=float, default=20.0, help="a foot carries load above this vertical force, N")
    args = p.parse_args()
    out = args.out or args.clip.with_suffix("")

    model = mujoco.MjModel.from_xml_path(str(args.xml))
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT | mujoco.mjtDisableBit.mjDSBL_LIMIT
    data = mujoco.MjData(model)
    c = np.load(args.clip, allow_pickle=True)
    qpos, rate = c["qpos"], float(c["fps"])
    T, dt, nv = len(qpos), 1.0 / float(c["fps"]), model.nv
    if qpos.shape[1] != model.nq:
        raise SystemExit(f"the clip has {qpos.shape[1]} coordinates, the model {model.nq}")
    mass, g = float(model.body_mass.sum()), 9.81

    qvel = np.zeros((T, nv))
    for t in range(1, T - 1):
        mujoco.mj_differentiatePos(model, qvel[t], 2 * dt, qpos[t - 1], qpos[t + 1])
    qvel[0], qvel[-1] = qvel[1], qvel[-2]
    qvel = lowpass(qvel, args.cutoff, rate)
    qacc = np.gradient(qvel, dt, axis=0)
    grf, cop, free = lowpass(c["grf"], args.cutoff, rate), c["cop"], lowpass(c["free_moment"], args.cutoff, rate)
    feet = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("calcn_r", "calcn_l")]

    tau, applied = np.zeros((T, nv)), np.zeros(nv)
    for t in range(T):
        data.qpos[:], data.qvel[:], data.qacc[:] = qpos[t], qvel[t], qacc[t]
        mujoco.mj_inverse(model, data)
        external = np.zeros(nv)
        for k, body in enumerate(feet):
            if grf[t, k, 2] > args.min_force:
                applied[:] = 0.0
                mujoco.mj_applyFT(model, data, grf[t, k], free[t, k], cop[t, k], body, applied)
                external += applied
        tau[t] = data.qfrc_inverse - external

    # ---------------------------------------------------------------- residual on the free root
    interior = np.zeros(T, bool); interior[10:-10] = True
    # A frame is usable when every foot that carries load has a usable centre of pressure.  (A foot in the
    # air has no centre of pressure; requiring one for both feet would keep double support only.)
    loaded = grf[:, :, 2] > args.min_force
    cop_ok = np.asarray(c["cop_frame_ok"], bool) if "cop_frame_ok" in c.files else np.ones((T, 2), bool)
    cop_ok = cop_ok if cop_ok.ndim == 2 else np.repeat(cop_ok[:, None], 2, axis=1)
    ok = interior & c["grf_frame_ok"] & ~(loaded & ~cop_ok[:, :2]).any(1)
    height = float(qpos[:, 2].mean())
    force, moment = np.linalg.norm(tau[:, :3], axis=1), np.linalg.norm(tau[:, 3:6], axis=1)
    print(f"[id] {args.clip.name} on {args.xml.name}: {T} frames at {rate:.1f} Hz, model {mass:.1f} kg, clip mass {float(c['mass_kg_dynamics']):.1f} kg, low-pass {args.cutoff:g} Hz")
    print(f"     frames used: {ok.mean():.0%} (interior, GRF usable, centre of pressure usable on every loaded foot); double support {np.mean(loaded.all(1)):.0%}, flight {np.mean(~loaded.any(1)):.0%}")
    print(f"     mean total vertical GRF / weight: {(grf[ok, :, 2].sum(1) / (mass * g)).mean():.3f}")
    print(f"     root residual, mean (95th percentile): force {force[ok].mean():.0f} ({np.percentile(force[ok], 95):.0f}) N = {100 * force[ok].mean() / (mass * g):.1f} % of weight;"
          f" moment {moment[ok].mean():.0f} ({np.percentile(moment[ok], 95):.0f}) N m = {100 * moment[ok].mean() / (mass * g * height):.1f} % of weight x pelvis height")

    length_scale = float(c["length_scale"]) if "length_scale" in c.files else 1.0
    force_scale = float(c["force_scale"]) if "force_scale" in c.files else 1.0
    # ForceBody reports its own residual in the first six torque columns (three moments, three forces).
    fb_force = np.linalg.norm(lowpass(c["tau"][:, 3:6], args.cutoff, rate), axis=1) * force_scale
    fb_moment = np.linalg.norm(lowpass(c["tau"][:, 0:3], args.cutoff, rate), axis=1) * force_scale * length_scale
    both = ok & c["tau_frame_ok"]
    if both.any():
        print(f"     ForceBody's own root residual on the same frames: force {fb_force[both].mean():.0f} N = {100 * fb_force[both].mean() / (mass * g):.1f} % of weight;"
              f" moment {fb_moment[both].mean():.0f} N m = {100 * fb_moment[both].mean() / (mass * g * height):.1f} %")
    if length_scale != 1.0 or force_scale != 1.0:
        print(f"     rescaled clip: lengths x {length_scale:.3f}, forces x {force_scale:.3f}, time unchanged")

    # ---------------------------------------------------------------- joint torques against the labels
    pairs = json.loads(str(c["label_pairs"])) if "label_pairs" in c.files else []
    jname = [model.joint(j).name for j in range(1, model.njnt)]
    pose_names = [str(n) for n in c["pose_names"]]
    # Labels are in N m for the subject.  For a rescaled clip they are brought to the model with the same
    # factors as the measured forces (mass ratio x leg-length ratio), then divided by the model mass.
    label_mass = mass / (force_scale * length_scale) if "force_scale" in c.files else float(c["mass_kg_dynamics"])
    use = ok & c["tau_frame_ok"]
    sig_ok = use & c["mc_frame_ok"]
    rows, series = [], {}
    print(f"     {'coordinate':20s} {'RMSE':>7s} {'range(label)':>13s} {'r':>6s} {'mean sigma':>11s} {'within 2 sigma':>15s}   (N m / kg; frames with labels: {use.mean():.0%})")
    for pair in pairs:
        if pair["skel"] not in pose_names:
            continue
        dof = 6 + pose_names.index(pair["skel"])
        mine = tau[:, dof] / mass
        label = lowpass(pair["sign"] * c["tau"][:, pair["column"]] / label_mass, args.cutoff, rate)
        sigma = c["sigma_tau"][:, pair["column"]] / label_mass
        series[pair["skel"]] = (mine, label, sigma)
        if use.sum() < 10:
            continue
        r = float(np.corrcoef(mine[use], label[use])[0, 1]) if label[use].std() > 1e-9 else float("nan")
        row = dict(name=pair["skel"], rmse=float(np.sqrt(np.mean((mine - label)[use] ** 2))), rng=float(np.ptp(label[use])), r=r,
                   sigma=float(sigma[sig_ok].mean()) if sig_ok.any() else float("nan"),
                   within=float(np.mean(np.abs(mine - label)[sig_ok] <= 2 * sigma[sig_ok])) if sig_ok.any() else float("nan"))
        rows.append(row)
        print(f"     {row['name']:20s} {row['rmse']:7.3f} {row['rng']:13.3f} {row['r']:6.2f} {row['sigma']:11.3f} {row['within']:15.0%}")

    time = np.arange(T) * dt
    shown = [n for n in ("hip_flexion_{s}", "hip_adduction_{s}", "hip_rotation_{s}", "knee_angle_{s}", "ankle_angle_{s}", "subtalar_angle_{s}")]
    fig, axes = plt.subplots(len(shown) + 1, 2, figsize=(13, 2.3 * (len(shown) + 1)), sharex=True)
    for col, side in enumerate("rl"):
        for row, pattern in enumerate(shown):
            ax, name = axes[row, col], pattern.format(s=side)
            if name in series:
                mine, label, sigma = series[name]
                ax.fill_between(time, label - 2 * sigma, label + 2 * sigma, color="0.85", label="label ± 2 sigma")
                ax.plot(time, label, color="k", lw=1.2, label="ForceBody label")
                ax.plot(time, mine, color="tab:blue", lw=1.2, label="inverse dynamics on the SKEL model")
            ax.set_title(name, fontsize=9, loc="left"); ax.set_ylabel("N m / kg"); ax.grid(alpha=0.3)
        axes[0, col].legend(fontsize=7, ncol=3)
        ax = axes[-1, col]
        ax.plot(time, 100 * force / (mass * g), color="tab:red", label="root residual force, % of weight")
        ax.plot(time, 100 * moment / (mass * g * height), color="tab:purple", label="root residual moment, % of weight x pelvis height")
        ax.set_xlabel("time (s)"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.suptitle(f"{args.clip.name}: inverse dynamics on {args.xml.name}", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{out}_id.png", dpi=120)
    np.savez_compressed(f"{out}_id.npz", tau=tau, qvel=qvel, qacc=qacc, joint_names=np.array(jname), time=time, ok=ok)
    summary = {
        "clip": str(args.clip), "model": str(args.xml), "frames": int(T), "frames_used": float(ok.mean()), "frames_with_labels": float(use.mean()),
        "model_mass_kg": mass, "length_scale": length_scale, "force_scale": force_scale,
        "vertical_grf_over_weight": float((grf[ok, :, 2].sum(1) / (mass * g)).mean()) if ok.any() else None,
        "residual_force_pct_weight": float(100 * force[ok].mean() / (mass * g)) if ok.any() else None,
        "residual_moment_pct": float(100 * moment[ok].mean() / (mass * g * height)) if ok.any() else None,
        "forcebody_residual_force_pct_weight": float(100 * fb_force[both].mean() / (mass * g)) if both.any() else None,
        "joints": rows,
    }
    Path(f"{out}_id.json").write_text(json.dumps(summary, indent=1))
    print(f"     saved {out}_id.png and {out}_id.npz")


if __name__ == "__main__":
    main()
