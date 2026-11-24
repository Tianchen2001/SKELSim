#!/usr/bin/env python3
"""Replay clips on a SKEL dynamics model and write videos (kinematic replay: mj_forward per frame, no simulation).

By default the body is drawn with the collision shapes of the model itself (capsules with gaps at the joints,
foot boxes), so the video shows what the simulation collides with; --shapes display draws connected capsules
instead.  No mesh files are needed.  The measured ground reaction force of the clip is
drawn as an arrow at the centre of pressure (1 m per body weight).  Right side blue, left side orange.
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
from lxml import etree
from PIL import Image, ImageDraw

RADIUS = {"pelvis": 0.095, "lumbar_body": 0.105, "thorax": 0.06, "scapula": 0.045, "humerus": 0.042, "ulna": 0.034, "radius": 0.03,
          "femur": 0.065, "tibia": 0.046, "talus": 0.035, "calcn": 0.03}
TRUNK, RIGHT, LEFT = "0.78 0.74 0.68 1", "0.30 0.47 0.70 1", "0.85 0.52 0.25 1"


def capsule_model(xml_path: Path, width: int, height: int, shapes: str = "collision") -> mujoco.MjModel:
    root = etree.parse(str(xml_path)).getroot()
    for asset in root.findall("asset"):
        root.remove(asset)
    asset = etree.SubElement(root, "asset")
    etree.SubElement(asset, "texture", type="skybox", builtin="gradient", rgb1="0.93 0.95 0.98", rgb2="0.75 0.80 0.88", width="64", height="64")
    etree.SubElement(asset, "texture", name="grid", type="2d", builtin="checker", rgb1="0.88 0.88 0.86", rgb2="0.80 0.80 0.78", width="256", height="256")
    etree.SubElement(asset, "material", name="grid", texture="grid", texrepeat="2 2", texuniform="true", reflectance="0.05")
    visual = etree.SubElement(root, "visual")
    etree.SubElement(visual, "global", offwidth=str(width), offheight=str(height))
    etree.SubElement(visual, "headlight", ambient="0.45 0.45 0.45", diffuse="0.5 0.5 0.5", specular="0 0 0")
    floor = root.find("worldbody/geom[@name='floor']")
    floor.set("material", "grid"); floor.set("rgba", "1 1 1 1")
    for light in root.iter("light"):
        light.set("castshadow", "false")  # shadows on a 100 m floor flicker; the feet show the contact well enough
    for contact in root.findall("contact"):
        root.remove(contact)
    for body in root.iter("body"):
        name = body.get("name")
        base = name[:-2] if name.endswith(("_r", "_l")) else name
        color = RIGHT if name.endswith("_r") else LEFT if name.endswith("_l") else TRUNK
        keep = ("foot", "collision") if shapes == "collision" else ("foot",)
        for geom in body.findall("geom"):
            if geom.get("class") not in keep:
                body.remove(geom)
            else:
                geom.set("rgba", color); geom.set("group", "0")
        if shapes == "collision":      # show exactly the shapes the simulation collides with
            continue
        add = lambda **kw: etree.SubElement(body, "geom", contype="0", conaffinity="0", rgba=color, **kw)
        children = [(c.get("name"), np.array(c.get("pos").split(), float)) for c in body.findall("body")]
        own = np.array(body.get("pos").split(), float)
        for child, pos in children:
            # Spine: the origin of lumbar_body, thorax and head is the TOP of the segment below it, so the trunk
            # volumes are drawn on the child, hanging down from its origin (see below), not on the parent.
            if child in ("lumbar_body", "thorax") or np.linalg.norm(pos) <= 0.03 or base not in RADIUS:
                continue
            radius = 0.05 if (base == "thorax" and child.startswith("scapula")) else RADIUS[base]
            add(type="capsule", fromto=" ".join(f"{v:.5f}" for v in (0, 0, 0, *pos)), size=f"{radius}")
        if base == "lumbar_body":   # abdomen: from the lumbar joint down towards the pelvis
            add(type="capsule", fromto=" ".join(f"{v:.5f}" for v in (0, 0, 0, *(-0.75 * own))), size="0.105")
        if base == "thorax":        # rib cage: from the base of the neck down to the lumbar joint
            add(type="capsule", fromto=" ".join(f"{v:.5f}" for v in (0, 0, -0.03, *(-0.9 * own))), size="0.115")
        if base == "pelvis":
            hips = dict(children)
            add(type="capsule", fromto=" ".join(f"{v:.5f}" for v in (*hips["femur_r"], *hips["femur_l"])), size="0.085")
        if base in ("head", "hand"):
            com = np.array(body.find("inertial").get("pos").split(), float)
            if base == "head":
                add(type="sphere", pos=" ".join(f"{v:.5f}" for v in com), size="0.105")
            else:
                add(type="capsule", fromto=" ".join(f"{v:.5f}" for v in (0, 0, 0, *(1.8 * com))), size="0.032")
    return mujoco.MjModel.from_xml_string(etree.tostring(root).decode())


def arrow(scene, start, end, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9), np.asarray(rgba, np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.02, np.asarray(start, float), np.asarray(end, float))
    scene.ngeom += 1


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--xml", required=True, type=Path)
    p.add_argument("--clips", required=True, nargs="+", help="clip npz files or glob patterns")
    p.add_argument("--out", required=True, type=Path, help="output directory")
    p.add_argument("--fps", type=float, default=50.0)
    p.add_argument("--seconds", type=float, default=8.0, help="longest video; 0 = whole clip")
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--azimuth", type=float, default=110.0, help="90 = from the left side, 180 = from the front")
    p.add_argument("--shapes", choices=["collision", "display"], default="collision",
                   help="collision: the collision shapes of the model as they are (gaps at the joints); display: connected capsules for looks")
    p.add_argument("--no-forces", action="store_true")
    args = p.parse_args()

    model = capsule_model(args.xml, args.width, args.height, args.shapes)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    weight = float(model.body_mass.sum()) * 9.81
    cam = mujoco.MjvCamera()
    cam.type, cam.distance, cam.elevation, cam.azimuth = mujoco.mjtCamera.mjCAMERA_FREE, 3.4, -8.0, args.azimuth
    args.out.mkdir(parents=True, exist_ok=True)
    files = sorted(f for pattern in args.clips for f in glob.glob(pattern))
    if not files:
        raise SystemExit("no clip matches")
    for file in files:
        c = np.load(file, allow_pickle=True)
        qpos, rate = c["qpos"], float(c["fps"])
        if qpos.shape[1] != model.nq:
            print(f"[skip] {file}: {qpos.shape[1]} coordinates, the model has {model.nq}")
            continue
        step = max(1, int(round(rate / args.fps)))
        frames = range(0, len(qpos) if args.seconds <= 0 else min(len(qpos), int(args.seconds * rate)), step)
        scale = f"lengths x{float(c['length_scale']):.3f}, forces x{float(c['force_scale']):.3f}" if "length_scale" in c.files else "not rescaled"
        target = args.out / (Path(file).stem.replace("_clip", "") + ".mp4")
        lookat = np.array([np.median(qpos[:, 0]), np.median(qpos[:, 1]), 0.95])
        with imageio.get_writer(str(target), fps=rate / step, codec="libx264", quality=8, macro_block_size=None) as writer:
            for t in frames:
                data.qpos[:] = qpos[t]
                mujoco.mj_forward(model, data)
                cam.lookat[:] = [qpos[t, 0], qpos[t, 1], lookat[2]]  # follow the pelvis; the floor grid shows the travel
                renderer.update_scene(data, camera=cam)
                if not args.no_forces and "grf" in c.files:
                    for k, color in enumerate(((0.1, 0.3, 0.8, 1), (0.9, 0.4, 0.1, 1))):
                        if c["grf"][t, k, 2] > 20.0:
                            arrow(renderer.scene, c["cop"][t, k], c["cop"][t, k] + c["grf"][t, k] / weight, color)
                image = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(image)
                draw.text((12, 10), f"{Path(file).stem.replace('_clip', '')}   t = {t / rate:5.2f} s", fill=(20, 20, 20))
                draw.text((12, 26), f"general SKEL model {float(model.body_mass.sum()):.1f} kg, {scale}; arrows: measured GRF (1 m = body weight)", fill=(60, 60, 60))
                writer.append_data(np.asarray(image))
        print(f"[saved] {target} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
