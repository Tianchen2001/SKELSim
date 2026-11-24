#!/usr/bin/env python3
"""Inverse dynamics on one SKEL dynamics model for many ForceBody trials (one per subject), with a summary.

Trials come from a list file (one npz path per line).  Only subjects fitted with the SKEL gender of the model
are used.  Each trial is rescaled to the model (leg length, mass), converted with the dynamics-pass poses and
run through skel_inverse_dynamics.py; the per-trial results are collected into one table and a CSV.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MAIN = ("hip_flexion", "hip_adduction", "hip_rotation", "knee_angle", "ankle_angle", "subtalar_angle")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", required=True, type=Path, help="text file with one trial npz per line")
    p.add_argument("--data-root", required=True, type=Path, help="directory the paths in the list are relative to")
    p.add_argument("--xml", required=True, type=Path)
    p.add_argument("--skel-model-path", required=True)
    p.add_argument("--out", required=True, type=Path, help="output directory")
    p.add_argument("--max-subjects", type=int, default=0, help="0 = all")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true", help="convert again even when the clip file exists")
    args = p.parse_args()

    gender = json.loads(args.xml.with_suffix(".json").read_text())["gender"]
    by_subject: dict[tuple[str, str], list[Path]] = {}
    for line in args.list.read_text().split():
        path = args.data_root / line
        by_subject.setdefault((path.parts[-3], path.parts[-2]), []).append(path)
    rng = random.Random(args.seed)
    chosen = []
    for key in sorted(by_subject):
        trial = rng.choice(sorted(by_subject[key]))
        trial_gender = "female" if "female" in str(np.load(trial, allow_pickle=True)["skel_gender"]).lower() else "male"
        if trial_gender == gender:
            chosen.append((key, trial))
    if args.max_subjects:
        rng.shuffle(chosen); chosen = sorted(chosen[: args.max_subjects])
    print(f"[batch] {len(chosen)} {gender} subjects from {len(by_subject)} in the list")
    args.out.mkdir(parents=True, exist_ok=True)

    results = []
    for (study, subject), trial in chosen:
        clip = args.out / f"{trial.stem}_clip.npz"
        if args.force or not clip.exists():
            run = subprocess.run([sys.executable, "-W", "ignore", str(HERE / "forcebody_to_skel_clip.py"), "--npz", str(trial), "--xml", str(args.xml),
                                  "--skel-model-path", args.skel_model_path, "--rescale", "--out", str(clip)], capture_output=True, text=True)
            if run.returncode != 0:
                print(f"[batch] {trial.stem}: conversion failed: {run.stderr.strip().splitlines()[-1] if run.stderr.strip() else 'no message'}")
                continue
        run = subprocess.run([sys.executable, "-W", "ignore", str(HERE / "skel_inverse_dynamics.py"), "--xml", str(args.xml), "--clip", str(clip)], capture_output=True, text=True)
        summary_path = clip.with_name(clip.stem + "_id.json")
        if run.returncode != 0 or not summary_path.exists():
            print(f"[batch] {trial.stem}: inverse dynamics failed: {run.stderr.strip().splitlines()[-1] if run.stderr.strip() else 'no message'}")
            continue
        s = json.loads(summary_path.read_text())
        s.update(study=study, subject=subject, trial=trial.stem)
        results.append(s)
        joint = {j["name"]: j for j in s["joints"]}
        knee = [joint[f"knee_angle_{x}"]["r"] for x in "rl" if f"knee_angle_{x}" in joint]
        print(f"[batch] {trial.stem:34s} length x{s['length_scale']:.3f} mass x{s['force_scale']:.3f}  frames {s['frames_with_labels']:4.0%}  "
              f"residual force {s['residual_force_pct_weight']:5.1f} %  moment {s['residual_moment_pct']:5.1f} %  knee r {np.mean(knee) if knee else float('nan'):.2f}")

    if not results:
        raise SystemExit("no trial finished")
    stat = lambda v: (float(np.median(v)), float(np.min(v)), float(np.max(v)))
    print(f"\n[summary] {len(results)} trials on {args.xml}")
    for key, label in (("length_scale", "length scale"), ("force_scale", "mass scale"), ("residual_force_pct_weight", "residual force, % of weight"),
                       ("residual_moment_pct", "residual moment, %"), ("forcebody_residual_force_pct_weight", "ForceBody residual force, %")):
        values = [r[key] for r in results if r.get(key) is not None]
        if values:
            print(f"   {label:30s} median {stat(values)[0]:6.2f}   range {stat(values)[1]:6.2f} to {stat(values)[2]:6.2f}")
    print(f"   {'coordinate (both sides)':26s} {'r median':>9s} {'r min':>7s} {'r max':>7s} {'RMSE median':>12s} {'RMSE max':>9s} {'within 2 sigma, median':>23s}   (N m / kg)")
    rows_csv = []
    for name in MAIN:
        js = [j for r in results for j in r["joints"] if j["name"] in (f"{name}_r", f"{name}_l") and np.isfinite(j["r"])]
        if not js:
            continue
        r_, e_, w_ = stat([j["r"] for j in js]), stat([j["rmse"] for j in js]), stat([j["within"] for j in js if np.isfinite(j["within"])] or [float("nan")])
        print(f"   {name:26s} {r_[0]:9.2f} {r_[1]:7.2f} {r_[2]:7.2f} {e_[0]:12.3f} {e_[2]:9.3f} {w_[0]:23.0%}")
    with open(args.out / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["study", "subject", "trial", "length_scale", "force_scale", "frames_with_labels", "residual_force_pct_weight", "residual_moment_pct", "coordinate", "r", "rmse", "range_label", "sigma", "within_2_sigma"])
        for r in results:
            for j in r["joints"]:
                w.writerow([r["study"], r["subject"], r["trial"], r["length_scale"], r["force_scale"], r["frames_with_labels"], r["residual_force_pct_weight"], r["residual_moment_pct"],
                            j["name"], j["r"], j["rmse"], j["rng"], j["sigma"], j["within"]])
    print(f"[saved] {args.out / 'summary.csv'}")


if __name__ == "__main__":
    main()
