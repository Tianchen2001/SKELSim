#!/usr/bin/env python3
"""Turn a kinematic SKEL MJCF (generate_skel_xml.py) into a model for dynamics.

Input: skel_humanoid.xml and skel_humanoid.mapping.json (the mapping must contain "bodies", i.e. be
written after the frame-export change).  Output: a new MJCF and a small JSON description.

What changes
  frame      x forward, y left, z up (the kinematic model is x left, y up, z forward).
  inertia    explicit <inertial> for every body.  Legs, pelvis and arms use the Rajagopal table that the
             ForceBody inverse dynamics uses: fixed mass fractions of the total mass, centre of mass at the
             unscaled OpenSim offset, inertia scaled with mass only.  The single Rajagopal torso is split
             over lumbar_body, thorax, head and the scapulae so that the total torso mass, its centre of
             mass and its inertia about that centre are reproduced in the zero pose.
  locked     head (3), wrists (4) at 0; scapula upward rotation at the constant found in ForceBody.
  contact    feet: one box under the rear foot (on calcn) and one under the toes, built from the sole of the
             skin meshes.  Rest of the body: capsules (a sphere for the head) that are shorter than the
             segments, so that neighbouring segments leave a gap at the joint, as in SMPLSim.  They collide
             with the floor and, unless --no-self-collision is given, with each other; pairs that sit next to
             each other across a joint are excluded.  Mass and inertia do not depend on these shapes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from lxml import etree
from scipy.spatial.transform import Rotation

# Old model frame (x left, y up, z forward) -> new frame (x forward, y left, z up).
R = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

# Rajagopal bodies as scaled by AddBiomechanics for ForceBody (identical for every subject): mass as a
# fraction of the total, centre of mass in the OpenSim body frame (m), principal inertia per kg of TOTAL mass.
RAJAGOPAL = {
    "pelvis": (0.156324, (-0.0707, 0.0, 0.0), (0.00136453, 0.00115614, 0.00076855)),
    "femur": (0.123464, (0.0, -0.17, 0.0), (0.00177735, 0.00046591, 0.00187424)),
    "tibia": (0.049212, (0.0, -0.1867, 0.0), (0.00066899, 0.0000677, 0.00067828)),
    "patella": (0.001145, (0.0018, 0.0264, 0.0), (4e-08, 1.7e-07, 1.7e-07)),
    "talus": (0.001327, (0.0, 0.0, 0.0), (1.327e-05, 1.327e-05, 1.327e-05)),
    "calcn": (0.016592, (0.1, 0.03, 0.0), (1.858e-05, 5.177e-05, 5.442e-05)),
    "toes": (0.002875, (0.0346, 0.006, -0.0175), (8.37e-06, 9.7e-06, 1.327e-05)),
    "torso": (0.356088, (-0.03, 0.32, 0.0), (0.01957204, 0.01002826, 0.01899994)),
    "humerus": (0.026979, (0.0, -0.1645, 0.0), (0.00015857, 0.0000547, 0.00017799)),
    "ulna": (0.008064, (0.0, -0.12052, 0.0), (0.00003932, 0.0000082, 0.00004265)),
    "radius": (0.008064, (0.0, -0.12052, 0.0), (0.00003932, 0.0000082, 0.00004265)),
    "hand": (0.006073, (0.0, -0.0681, 0.0), (0.00001184, 0.00000726, 0.00001779)),
}
BACK_JOINT_IN_PELVIS = np.array([-0.1007, 0.0815, 0.0])  # generic Rajagopal, OpenSim pelvis frame
TORSO_PARTS = ("lumbar_body", "thorax", "head", "scapula_r", "scapula_l")
# Split of the torso (fractions of TOTAL body mass); the thorax takes what is left.
HEAD_FRACTION, SCAPULA_FRACTION, LUMBAR_SHARE_OF_TORSO = 0.0694, 0.005, 0.15

LOCKED_AT_ZERO = ("head_x", "head_y", "head_z", "hand_r_x", "hand_r_y", "hand_l_x", "hand_l_y")
LOCKED_AT = {"scapula_r_x": -0.190, "scapula_l_x": -0.210}  # rad, constant in the ForceBody SKEL fits
RANGE_FIX_DEG = {"scapula_r_y": (5.7296, 22.9183)}  # the right side is positive in the ForceBody fits

# Collision capsules: radius (m) and the part of the segment they cover (0 = own joint, 1 = next joint).
COLLISION = {"femur": (0.058, 0.16, 0.86), "tibia": (0.040, 0.14, 0.84), "humerus": (0.036, 0.16, 0.86), "radius": (0.028, 0.10, 0.88)}
# Body pairs that are not parent and child but still sit across a joint from each other.
EXCLUDE = [("pelvis", "thorax"), ("lumbar_body", "head")]
EXCLUDE += [pair for s in "rl" for pair in (
    ("thorax", f"humerus_{s}"), ("head", f"humerus_{s}"), ("lumbar_body", f"femur_{s}"), ("humerus_{s}".format(s=s), f"radius_{s}"),
    (f"tibia_{s}", f"calcn_{s}"), (f"tibia_{s}", f"toes_{s}"), (f"humerus_{s}", f"hand_{s}"))]


def vec(text: str) -> np.ndarray:
    return np.array(text.split(), dtype=np.float64)


def fmt(values) -> str:
    return " ".join(f"{float(v):.6g}" for v in np.asarray(values).ravel())


def quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    x, y, z, w = Rotation.from_matrix(matrix).as_quat()
    return np.array([w, x, y, z])


def shifted(inertia: np.ndarray, mass: float, d: np.ndarray) -> np.ndarray:
    """Inertia of a body about a point displaced by d from its centre of mass."""
    return inertia + mass * (float(d @ d) * np.eye(3) - np.outer(d, d))


def read_stl(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    count = int(np.frombuffer(raw, dtype="<u4", count=1, offset=80)[0])
    if len(raw) != 84 + 50 * count:
        raise ValueError(f"{path} is not a binary STL")
    tri = np.frombuffer(raw, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), count=count, offset=84)
    return tri["v"].reshape(-1, 3).astype(np.float64)


def sole_box(verts: np.ndarray, slab: float = 0.03, thickness: float = 0.04):
    """Box under the lowest `slab` metres of a foot mesh (old frame, y up): centre, half sizes."""
    low = verts[verts[:, 1] < verts[:, 1].min() + slab]
    lo, hi = low.min(0), low.max(0)
    lo[1], hi[1] = verts[:, 1].min(), verts[:, 1].min() + thickness
    return (lo + hi) / 2, (hi - lo) / 2


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--xml", required=True, type=Path)
    p.add_argument("--mapping", required=True, type=Path)
    p.add_argument("--mass", type=float, default=None, help="total mass in kg (default: mass_kg of the mapping)")
    p.add_argument("--out", required=True, type=Path, help="output MJCF")
    p.add_argument("--no-visual-meshes", action="store_true", help="leave the skin meshes out (collision boxes stay)")
    p.add_argument("--no-self-collision", action="store_true", help="body shapes collide with the floor only, not with each other")
    args = p.parse_args()

    mapping = json.loads(args.mapping.read_text())
    if "bodies" not in mapping:
        raise SystemExit("the mapping has no 'bodies' entry; regenerate it with the current generate_skel_xml.py")
    total = float(args.mass if args.mass is not None else mapping["mass_kg"])
    info = {b["name"]: b for b in mapping["bodies"]}
    origin = {n: np.array(b["origin_model_frame"]) for n, b in info.items()}
    frame = {n: np.array(b["opensim_frame_in_model_frame"]) for n, b in info.items()}  # columns: OpenSim x, y, z
    pose_of = {e["mjcf_joint_name"]: e for e in mapping["internal_coordinates"]}

    src = etree.parse(str(args.xml)).getroot()
    mesh_file = {m.get("name"): args.xml.parent / m.get("file") for m in src.iter("mesh")}

    # ---------------------------------------------------------------- inertia, old frame, body coordinates
    inertial: dict[str, tuple[float, np.ndarray, np.ndarray]] = {}
    for name in info:
        base = name[:-2] if name.endswith(("_r", "_l")) else name
        if base in RAJAGOPAL and name not in TORSO_PARTS:
            frac, com, diag = RAJAGOPAL[base]
            mass, c = frac * total, frame[name]
            inertia, com = c @ np.diag(diag) @ c.T * total, c @ np.array(com)
            if base == "femur":  # the patella rides on the femur; add it as a point mass at its own position
                pf, pc, _ = RAJAGOPAL["patella"]
                pm, pc = pf * total, c @ (np.array([-0.0045, -0.4096, 0.0]) + np.array(pc))
                new_com = (mass * com + pm * pc) / (mass + pm)
                inertia = shifted(inertia, mass, com - new_com) + shifted(np.zeros((3, 3)), pm, pc - new_com)
                mass, com = mass + pm, new_com
            inertial[name] = (mass, com, inertia)

    # Torso: reproduce the Rajagopal torso (mass, centre of mass, inertia) with the SKEL trunk bodies.
    cp = frame["pelvis"]
    tf, tcom, tdiag = RAJAGOPAL["torso"]
    t_mass = tf * total
    t_com = origin["pelvis"] + cp @ (BACK_JOINT_IN_PELVIS + np.array(tcom))  # model frame, zero pose
    t_inertia = cp @ np.diag(tdiag) @ cp.T * total
    lumbar_len = abs(origin["lumbar_body"][1] - origin["pelvis"][1])
    parts = {  # mass, centre of mass in the model frame, inertia about it
        "lumbar_body": (LUMBAR_SHARE_OF_TORSO * t_mass, origin["lumbar_body"] - np.array([0.0, lumbar_len / 2, 0.0]), None),
        "head": (HEAD_FRACTION * total, origin["head"] + np.array([0.0, 0.09, 0.02]), None),
        "scapula_r": (SCAPULA_FRACTION * total, origin["scapula_r"].copy(), None),
        "scapula_l": (SCAPULA_FRACTION * total, origin["scapula_l"].copy(), None),
    }
    box = lambda m, x, y, z: m / 12.0 * np.diag([y * y + z * z, x * x + z * z, x * x + y * y])
    sphere = lambda m, r: 0.4 * m * r * r * np.eye(3)
    parts["lumbar_body"] = (*parts["lumbar_body"][:2], box(parts["lumbar_body"][0], 0.30, lumbar_len, 0.20))
    parts["head"] = (*parts["head"][:2], sphere(parts["head"][0], 0.09))
    for side in "rl":
        parts[f"scapula_{side}"] = (*parts[f"scapula_{side}"][:2], sphere(parts[f"scapula_{side}"][0], 0.05))
    th_mass = t_mass - sum(m for m, _, _ in parts.values())
    th_com = (t_mass * t_com - sum(m * c for m, c, _ in parts.values())) / th_mass
    th_inertia = t_inertia - sum(shifted(i, m, c - t_com) for m, c, i in parts.values())
    th_inertia = th_inertia - (shifted(np.zeros((3, 3)), th_mass, th_com - t_com))
    eig = np.linalg.eigvalsh(th_inertia)
    if eig[0] <= 0 or eig[0] + eig[1] < eig[2]:
        raise SystemExit(f"the torso split leaves a non-physical thorax inertia (eigenvalues {eig}); change the split")
    parts["thorax"] = (th_mass, th_com, th_inertia)
    for name, (m, c, i) in parts.items():
        inertial[name] = (m, c - origin[name], i)

    # ---------------------------------------------------------------- feet: sole boxes from the skin meshes
    feet, floor_y = {}, []
    for side in "rl":
        try:
            talus = read_stl(mesh_file[f"talus_{side}_mesh"]) + origin[f"talus_{side}"] - origin[f"calcn_{side}"]
            toes = read_stl(mesh_file[f"toes_{side}_mesh"])
            source = "skin meshes"
        except (FileNotFoundError, KeyError, ValueError) as error:
            raise SystemExit(f"cannot read the foot meshes next to {args.xml} ({error})")
        for body, verts in ((f"calcn_{side}", talus), (f"toes_{side}", toes)):
            feet[body] = sole_box(verts)
            floor_y.append(origin[body][1] + verts[:, 1].min())
    height = -min(floor_y)

    # ---------------------------------------------------------------- write the new model
    root = etree.Element("mujoco", model="skel_dynamics")
    etree.SubElement(root, "compiler", angle="radian", autolimits="true")
    etree.SubElement(root, "option", timestep="0.002")
    default = etree.SubElement(root, "default")
    etree.SubElement(default, "joint", armature="0.01", damping="0")
    etree.SubElement(default, "geom", contype="0", conaffinity="0", rgba="0.8 0.6 0.4 1", group="1")
    # contype 0 / conaffinity 1 collides with the floor (contype 1) but not with another such geom.
    own_type = "0" if args.no_self_collision else "1"
    foot_default = etree.SubElement(default, "default", **{"class": "foot"})
    etree.SubElement(foot_default, "geom", contype=own_type, conaffinity="1", condim="3", friction="1 0.005 0.0001", rgba="0.3 0.5 0.8 0.6", group="0")
    body_default = etree.SubElement(default, "default", **{"class": "collision"})
    etree.SubElement(body_default, "geom", contype=own_type, conaffinity="1", condim="3", friction="1 0.005 0.0001", rgba="0.5 0.7 0.5 0.35", group="3")
    asset = etree.SubElement(root, "asset")
    if not args.no_visual_meshes:
        for name, path in mesh_file.items():
            etree.SubElement(asset, "mesh", name=name, file=str(path.relative_to(args.xml.parent)))
    world = etree.SubElement(root, "worldbody")
    etree.SubElement(world, "light", pos="0 0 3", dir="0 0 -1", directional="true")
    etree.SubElement(world, "geom", name="floor", type="plane", size="50 50 0.1", contype="1", conaffinity="1", condim="3", friction="1 0.005 0.0001", rgba="0.8 0.9 0.8 1", group="0")
    actuators, joints_out, locked_out = etree.Element("actuator"), [], {}
    mesh_quat = fmt(quat_wxyz(R))

    def convert(body, parent):
        name = body.get("name")
        pos = vec(body.get("pos"))
        attrib = {"name": name, "pos": fmt(R @ pos if parent is not world else [0.0, 0.0, height])}
        baked = np.eye(3)
        kept = []
        for joint in body.findall("joint"):
            jname = joint.get("name")
            if jname in LOCKED_AT_ZERO:
                locked_out[pose_of[jname]["pose_name"]] = 0.0
            elif jname in LOCKED_AT:
                if kept or np.linalg.norm(vec(joint.get("pos"))) > 1e-9:
                    raise SystemExit(f"{jname} cannot be baked: it is not the first joint or its anchor is off the origin")
                baked = baked @ Rotation.from_rotvec(LOCKED_AT[jname] * vec(joint.get("axis")) / np.linalg.norm(vec(joint.get("axis")))).as_matrix()
                locked_out[pose_of[jname]["pose_name"]] = LOCKED_AT[jname]
            else:
                kept.append(joint)
        if not np.allclose(baked, np.eye(3)):
            attrib["quat"] = fmt(quat_wxyz(R @ baked @ R.T))
        node = etree.SubElement(parent, "body", attrib)
        if parent is world:
            etree.SubElement(node, "freejoint", name=name)
        for joint in kept:
            jname, axis = joint.get("name"), vec(joint.get("axis"))
            lo, hi = RANGE_FIX_DEG.get(jname, vec(joint.get("range")))
            user = vec(joint.get("user"))
            etree.SubElement(node, "joint", name=jname, type="hinge", pos=fmt(R @ vec(joint.get("pos"))), axis=fmt(R @ (axis / np.linalg.norm(axis))),
                             range=fmt(np.radians([lo, hi])))
            etree.SubElement(actuators, "motor", name=jname, joint=jname, gear="1", ctrlrange=fmt([-user[3], user[3]]))
            joints_out.append({"mjcf_joint_name": jname, "pose_index": pose_of[jname]["pose_index"], "pose_name": pose_of[jname]["pose_name"]})
        mass, com, inertia = inertial[name]
        full = R @ inertia @ R.T
        etree.SubElement(node, "inertial", pos=fmt(R @ com), mass=fmt([mass]),
                         fullinertia=fmt([full[0, 0], full[1, 1], full[2, 2], full[0, 1], full[0, 2], full[1, 2]]))
        for geom in body.findall("geom"):
            if geom.get("type") == "mesh" and not args.no_visual_meshes:
                etree.SubElement(node, "geom", type="mesh", mesh=geom.get("mesh"), quat=mesh_quat)
            elif geom.get("type") == "capsule":
                ends = vec(geom.get("fromto")).reshape(2, 3) @ R.T
                etree.SubElement(node, "geom", type="capsule", fromto=fmt(ends), size=geom.get("size"))
        base = name[:-2] if name.endswith(("_r", "_l")) else name
        kids = {c.get("name"): vec(c.get("pos")) for c in body.findall("body")}
        col = lambda **kw: etree.SubElement(node, "geom", {"class": "collision", "name": f"{name}_col{len(node.findall('geom'))}", **kw})
        seg = lambda a, b, r: col(type="capsule", fromto=fmt(np.concatenate([R @ a, R @ b])), size=fmt([r]))
        if base in COLLISION and kids:
            r, a, b = COLLISION[base]
            end = next(iter(kids.values()))
            seg(a * end, b * end, r)
        elif base == "pelvis":
            seg(0.55 * kids["femur_r"], 0.55 * kids["femur_l"], 0.085)
        elif base == "lumbar_body":      # abdomen, below the lumbar joint
            seg(-0.25 * pos, -0.65 * pos, 0.095)
        elif base == "thorax":           # rib cage, below the base of the neck
            seg(-0.12 * pos, -0.80 * pos, 0.105)
        elif base == "head":
            col(type="sphere", pos=fmt(R @ com), size="0.095")
        elif base == "hand":
            seg(0.5 * com, 1.9 * com, 0.028)
        if name in feet:
            centre, half = feet[name]
            etree.SubElement(node, "geom", {"class": "foot", "name": f"{name}_sole", "type": "box", "pos": fmt(R @ centre), "size": fmt(np.abs(R @ half))})
        for child in body.findall("body"):
            convert(child, node)

    convert(src.find("worldbody/body"), world)
    root.append(actuators)
    if not args.no_self_collision:
        contact = etree.SubElement(root, "contact")
        for a, b in EXCLUDE:
            etree.SubElement(contact, "exclude", body1=a, body2=b)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    etree.ElementTree(root).write(str(args.out), pretty_print=True, xml_declaration=True, encoding="utf-8")

    # ---------------------------------------------------------------- check and report
    model = mujoco.MjModel.from_xml_path(str(args.out))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    com = data.subtree_com[1]
    ankle = np.mean([data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"talus_{s}")] for s in "rl"], axis=0)
    print(f"[model] {args.out}: {model.nbody - 1} bodies, {model.nu} actuated hinges, {len(locked_out)} locked, total mass {model.body_mass.sum():.2f} kg")
    print(f"[model] zero pose: pelvis origin {height:.3f} m above the floor, centre of mass height {com[2]:.3f} m, "
          f"{1000 * (com[0] - ankle[0]):+.0f} mm in front of the ankles, {1000 * com[1]:+.0f} mm to the left")
    print(f"[model] feet from the {source}; rear-foot box {fmt(2 * np.abs(R @ feet['calcn_r'][1]))} m, toe box {fmt(2 * np.abs(R @ feet['toes_r'][1]))} m (length, width, height)")
    self_pairs = [(model.body(model.geom_bodyid[c.geom1]).name, model.body(model.geom_bodyid[c.geom2]).name, c.dist) for c in data.contact[: data.ncon]
                  if model.geom_bodyid[c.geom1] > 0 and model.geom_bodyid[c.geom2] > 0 and c.dist < 0]
    print(f"[model] collision: {sum(1 for g in range(model.ngeom) if model.geom_conaffinity[g] and model.geom_bodyid[g] > 0)} body shapes, self-collision "
          f"{'off' if args.no_self_collision else 'on'}; penetrating body pairs in the zero pose: {self_pairs if self_pairs else 'none'}")
    print(f"{'body':12s} {'mass kg':>8s} {'percent':>8s}   centre of mass in the body frame (forward, left, up), mm")
    for i in range(1, model.nbody):
        print(f"{model.body(i).name:12s} {model.body_mass[i]:8.3f} {100 * model.body_mass[i] / model.body_mass.sum():8.2f}   " + " ".join(f"{1000 * v:+7.1f}" for v in model.body_ipos[i]))
    description = {
        "source_xml": str(args.xml), "total_mass_kg": total, "frame": "x forward, y left, z up",
        "old_to_new_rotation": R.tolist(), "pelvis_height_zero_pose_m": height,
        "gender": mapping["gender"], "betas": mapping["betas"],
        "joints": joints_out, "locked_pose_values_rad": locked_out, "self_collision": not args.no_self_collision,
        "torso_target": {"mass_kg": t_mass, "com_model_frame_old": t_com.tolist()},
        "inertial_rule": "Rajagopal fractions of total mass, unscaled OpenSim centre of mass, inertia scaled with mass (as in the ForceBody models)",
    }
    args.out.with_suffix(".json").write_text(json.dumps(description, indent=2))
    print(f"[saved] {args.out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
