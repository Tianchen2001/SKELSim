"""SKEL-specific front end that reuses SMPLSim's geometry and MJCF writer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lxml import etree

import smpl_sim.smpllib.skeleton_mesh_local as skeleton_mesh_module
from smpl_sim.smpllib.smpl_local_robot import get_joint_geometries
from smpl_sim.smpllib.skel_joint_spec import (
    SKEL_BODY_NAMES,
    SKEL_GAINS_MJ,
    SKEL_PARENT_INDICES,
)
from smpl_sim.smpllib.skel_parser import SKEL_Parser


def _normalize_gender(value: object) -> str:
    text = str(value).strip().lower()
    if "female" in text or text == "f":
        return "female"
    return "male"


def _adjacent_contact_exclusions() -> list[list[str]]:
    """Exclude collisions between every directly connected body pair."""
    exclusions: list[list[str]] = []
    for child_index, parent_index in enumerate(SKEL_PARENT_INDICES):
        if parent_index >= 0:
            exclusions.append(
                [SKEL_BODY_NAMES[parent_index], SKEL_BODY_NAMES[child_index]]
            )
    return exclusions


class SKEL_Robot:
    """Generate one subject-specific SKEL MJCF with SMPLSim's mesh pipeline.

    The class deliberately does not inherit from ``SMPL_Robot`` because that
    class hard-codes SMPL/SMPL-H/SMPL-X pose dimensions, body names, limits,
    contact exclusions, and parser selection.  This smaller front end reuses
    only the generic pieces:
      - ``get_joint_geometries()``
      - ``SkeletonMesh.load_from_offsets()``
      - ``SkeletonMesh.write_xml()``
    """

    def __init__(
        self,
        *,
        skel_model_path: str,
        output_dir: str | Path,
        gender: str,
        axis_epsilon: float = 1e-4,
        reverse_multidof_order: bool = True,
        real_weight: bool = False,
        anchor_from_translation: bool = True,
    ) -> None:
        self.anchor_from_translation = bool(anchor_from_translation)
        self.anchor_report: list[tuple[str, float, float]] = []
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.asset_dir = self.output_dir / "skel_assets"
        self.geom_dir = self.asset_dir / "geom"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.geom_dir.mkdir(parents=True, exist_ok=True)

        self.gender = _normalize_gender(gender)
        self.real_weight = bool(real_weight)
        self.parser = SKEL_Parser(
            model_path=skel_model_path,
            gender=self.gender,
            axis_epsilon=axis_epsilon,
            reverse_multidof_order=reverse_multidof_order,
        )

        # SkeletonMesh uses module-level gain dictionaries.  Updating the
        # dictionary is enough to support SKEL names without editing the
        # upstream SMPLSim source file.  Original SMPL keys remain untouched.
        skeleton_mesh_module.GAINS_MJ.update(SKEL_GAINS_MJ)
        skeleton_mesh_module.GAINS_PHC.update(SKEL_GAINS_MJ)
        self.skeleton = skeleton_mesh_module.Skeleton(str(self.asset_dir))

        self.betas: np.ndarray | None = None
        self.mass_kg: float | None = None
        self.hull_dict: dict[str, Any] | None = None

    @classmethod
    def from_forcebody_npz(
        cls,
        npz_path: str | Path,
        *,
        skel_model_path: str,
        output_dir: str | Path,
        axis_epsilon: float = 1e-4,
        reverse_multidof_order: bool = True,
        real_weight: bool = False,
        anchor_from_translation: bool = True,
    ) -> "SKEL_Robot":
        data = np.load(npz_path, allow_pickle=True)
        robot = cls(
            skel_model_path=skel_model_path,
            output_dir=output_dir,
            # ``skel_gender`` is the SKEL model the poses were fitted with.
            # ``sex`` is the recorded sex and can be "unknown".
            gender=str(
                data["skel_gender"]
                if "skel_gender" in data.files
                else data["sex"] if "sex" in data.files else "male"
            ),
            axis_epsilon=axis_epsilon,
            reverse_multidof_order=reverse_multidof_order,
            real_weight=real_weight,
            anchor_from_translation=anchor_from_translation,
        )
        robot.betas = np.asarray(data["betas"], dtype=np.float32).reshape(10)
        robot.mass_kg = (
            float(data["mass_kg"]) if "mass_kg" in data.files else None
        )
        return robot

    def build(
        self,
        *,
        betas: np.ndarray | torch.Tensor | None = None,
        flatfoot: bool = False,
        verbose_geometry: bool = False,
    ) -> None:
        """Build convex body meshes and populate the SMPLSim skeleton writer."""
        if betas is None:
            if self.betas is None:
                betas_array = np.zeros(10, dtype=np.float32)
            else:
                betas_array = self.betas
        else:
            betas_array = np.asarray(betas, dtype=np.float32).reshape(10)
            self.betas = betas_array

        parser_output = self.parser.get_mesh_offsets(
            zero_pose=torch.zeros((1, 46), dtype=torch.float32),
            betas=torch.from_numpy(betas_array).reshape(1, 10),
            flatfoot=flatfoot,
        )
        (
            vertices,
            joints,
            skin_weights,
            joint_names,
            joint_offsets,
            joint_parents,
            joint_axes,
            joint_dofs,
            joint_range,
            contype,
            conaffinity,
        ) = parser_output

        self.hull_dict = get_joint_geometries(
            vertices,
            joints,
            skin_weights,
            joint_names,
            geom_dir=str(self.geom_dir),
            scale_dict={},
            verbose=verbose_geometry,
        )

        self.skeleton.load_from_offsets(
            joint_offsets,
            joint_parents,
            joint_axes,
            joint_dofs,
            joint_range,
            sites={},
            scale=1.0,
            equalities={},
            hull_dict=self.hull_dict,
            sim="mujoco",
            exclude_contacts=_adjacent_contact_exclusions(),
            collision_groups=contype,
            conaffinity=conaffinity,
            simple_geom=False,
            color_dict=None,
            real_weight=self.real_weight,
            replace_feet=False,
            upright_start=False,
            create_vel_sensors=False,
        )

    def write_xml(self, xml_path: str | Path) -> Path:
        """Write the subject MJCF using SMPLSim's original XML writer.

        Every physical body is created at its anatomical joint origin.  Hinge
        anchors must therefore be local ``0 0 0``.  Some SMPLSim revisions
        serialize the body offset again as the joint ``pos``; the post-pass
        below removes that duplicate offset without replacing the upstream
        writer.
        """
        xml_path = Path(xml_path).expanduser().resolve()
        xml_path.parent.mkdir(parents=True, exist_ok=True)
        self.skeleton.write_xml(str(xml_path))

        tree = etree.parse(str(xml_path))
        coupling = self.parser.last_translation_coupling
        self.anchor_report = []
        for joint_node in tree.xpath(
            "/mujoco/worldbody//joint[not(@type='free')]"
        ):
            # SKEL moves the origin of some bodies as their own joint angles
            # change (constant-curvature spine, scapula on the thorax
            # ellipsoid).  A hinge whose anchor is shifted off the body origin
            # produces exactly such a motion: with unit axis a and anchor c
            # (body frame, origin o = 0), d(origin)/dq = a x (o - c).  Choosing
            # c = a x d reproduces the SKEL derivative d = d(translation)/dq
            # at the zero pose, for the part of d perpendicular to a.
            anchor = np.zeros(3)
            if self.anchor_from_translation:
                axis = np.asarray(
                    joint_node.attrib["axis"].split(), dtype=np.float64
                )
                axis = axis / np.linalg.norm(axis)
                derivative = np.asarray(
                    coupling.get(joint_node.attrib["name"], np.zeros(3)),
                    dtype=np.float64,
                )
                if np.linalg.norm(derivative) > 1e-3:
                    anchor = np.cross(axis, derivative)
                    self.anchor_report.append(
                        (
                            joint_node.attrib["name"],
                            float(np.linalg.norm(derivative)),
                            float(abs(np.dot(axis, derivative))),
                        )
                    )
            joint_node.attrib["pos"] = "{:.6f} {:.6f} {:.6f}".format(*anchor)
        tree.write(
            str(xml_path),
            pretty_print=True,
            xml_declaration=True,
            encoding="utf-8",
        )
        return xml_path

    def write_mapping(self, mapping_path: str | Path) -> Path:
        """Write the ForceBody pose-index to MJCF-joint mapping."""
        mapping_path = Path(mapping_path).expanduser().resolve()
        mapping_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.parser.last_mapping:
            raise RuntimeError("Call build() before write_mapping().")

        mapping = {
            "format_version": 1,
            "model": "SKEL v1.1.1 through SMPLSim SkeletonMesh",
            "gender": self.gender,
            "betas": (
                None if self.betas is None else self.betas.astype(float).tolist()
            ),
            "mass_kg": self.mass_kg,
            "native_coordinates": "+X right, +Y up, +Z forward",
            "mujoco_alignment": (
                "Playback applies +90 degrees about world X, mapping "
                "(x, y, z) -> (x, -z, y)."
            ),
            "root": {
                "joint_name": "pelvis",
                "pose_indices": [0, 1, 2],
                "zero_root_skel": (
                    None
                    if self.parser.last_zero_root is None
                    else self.parser.last_zero_root.astype(float).tolist()
                ),
            },
            "internal_coordinates": [
                {
                    "body_name": item.body_name,
                    "pose_index": item.pose_index,
                    "pose_name": item.pose_name,
                    "channel": item.channel,
                    "mjcf_joint_name": item.mjcf_joint_name,
                    "axis_parent_frame": item.axis_parent_frame.astype(
                        float
                    ).tolist(),
                    "translation_derivative_m_per_rad": (
                        item.translation_derivative.astype(float).tolist()
                    ),
                    "translation_coupling_norm_m_per_rad": float(
                        np.linalg.norm(item.translation_derivative)
                    ),
                }
                for item in self.parser.last_mapping
            ],
            "bodies": [
                {
                    "name": name,
                    "parent": (
                        SKEL_BODY_NAMES[SKEL_PARENT_INDICES[index]]
                        if SKEL_PARENT_INDICES[index] >= 0
                        else None
                    ),
                    "origin_model_frame": self.parser.last_body_origins[
                        index
                    ].tolist(),
                    "opensim_frame_in_model_frame": self.parser.last_body_frames[
                        index
                    ].tolist(),
                }
                for index, name in enumerate(SKEL_BODY_NAMES)
            ],
            "known_kinematic_approximations": [
                "Scapular ellipsoid translation is not represented by static hinges.",
                "Constant-curvature spine translations are not represented by static hinges.",
                "The current public SKEL forward pass does not enable the commented Walker-knee translation.",
            ],
        }
        mapping_path.write_text(
            json.dumps(mapping, indent=2), encoding="utf-8"
        )
        return mapping_path

    def normalize_xml_mass(
        self,
        xml_path: str | Path,
        target_mass_kg: float | None = None,
    ) -> float:
        """Scale mesh densities so MuJoCo's compiled body mass matches a target.

        This is a convenience for later dynamics experiments.  It does not
        recover the original AddBiomechanics segment mass distribution; it
        only scales SMPLSim's volume-based distribution to the subject total.
        """
        import mujoco

        xml_path = Path(xml_path).expanduser().resolve()
        target = (
            float(target_mass_kg)
            if target_mass_kg is not None
            else self.mass_kg
        )
        if target is None or target <= 0:
            raise ValueError("A positive target mass is required.")

        model = mujoco.MjModel.from_xml_path(str(xml_path))
        current_mass = float(np.sum(model.body_mass))
        if current_mass <= 0:
            raise RuntimeError("The generated model has zero compiled mass.")

        scale = target / current_mass
        tree = etree.parse(str(xml_path))
        density_nodes = tree.xpath(
            "/mujoco/worldbody//geom[@density]"
        )
        if not density_nodes:
            raise RuntimeError(
                "No explicit body geom densities were found in the XML."
            )
        for node in density_nodes:
            node.attrib["density"] = str(
                float(node.attrib["density"]) * scale
            )
        tree.write(str(xml_path), pretty_print=True)

        verified = mujoco.MjModel.from_xml_path(str(xml_path))
        return float(np.sum(verified.body_mass))
