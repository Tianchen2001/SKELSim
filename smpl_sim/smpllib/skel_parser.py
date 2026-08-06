"""Adapter from official SKEL v1.1.1 outputs to SMPLSim's parser contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from skel.skel_model import SKEL

from smpl_sim.smpllib.skel_joint_spec import (
    SKEL_BODY_NAMES,
    SKEL_BODY_POSE_INDICES,
    SKEL_PARENT_INDICES,
    SKEL_POSE_NAMES,
    pose_limit,
)


@dataclass(frozen=True)
class JointCoordinateMapping:
    """Mapping between one ForceBody/SKEL pose scalar and one MJCF hinge."""

    body_name: str
    pose_index: int
    pose_name: str
    channel: str
    mjcf_joint_name: str
    axis_parent_frame: np.ndarray
    translation_derivative: np.ndarray


def _as_dense_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a potentially sparse torch tensor to a NumPy array."""
    if tensor.is_sparse:
        tensor = tensor.to_dense()
    return tensor.detach().cpu().numpy()


def _normalize_gender(gender: str) -> str:
    value = str(gender).strip().lower()
    if "female" in value or value == "f":
        return "female"
    if "male" in value or value == "m":
        return "male"
    raise ValueError(f"Unsupported SKEL gender: {gender!r}")


def _relative_rotation(
    global_orientations: np.ndarray,
    body_index: int,
    parent_index: int,
) -> np.ndarray:
    """Return the child orientation expressed relative to its parent."""
    return global_orientations[parent_index].T @ global_orientations[body_index]


def _complete_axis_matrix(active_axes: list[np.ndarray]) -> np.ndarray:
    """Pack up to three active axes into the columns of a 3x3 matrix.

    SkeletonMesh indexes columns through channel names x/y/z.  Unused columns
    have no kinematic effect, but are filled with stable unit vectors to avoid
    malformed matrices in debugging output.
    """
    if not 1 <= len(active_axes) <= 3:
        raise ValueError(f"Expected 1..3 axes, got {len(active_axes)}.")

    columns = [np.asarray(axis, dtype=np.float64) for axis in active_axes]
    basis = [
        np.asarray([1.0, 0.0, 0.0]),
        np.asarray([0.0, 1.0, 0.0]),
        np.asarray([0.0, 0.0, 1.0]),
    ]
    while len(columns) < 3:
        candidate = basis[len(columns)]
        columns.append(candidate)
    return np.column_stack(columns)


class SKEL_Parser:
    """Expose SKEL through the interface consumed by SMPLSim.

    The adapter keeps the generated MJCF in SKEL's native Y-up coordinates.
    The playback script applies one global +90-degree X rotation to place the
    character upright in MuJoCo's conventional Z-up world.
    """

    def __init__(
        self,
        model_path: str,
        gender: str,
        *,
        axis_epsilon: float = 1e-4,
        reverse_multidof_order: bool = True,
    ) -> None:
        self.gender = _normalize_gender(gender)
        self.model = SKEL(gender=self.gender, model_path=model_path)
        self.axis_epsilon = float(axis_epsilon)
        self.reverse_multidof_order = bool(reverse_multidof_order)

        self.joint_names = [str(name) for name in self.model.bone_names]

        # ``pose_params_name`` in the distributed SKEL pickle contains
        # 64 SMPL-style metadata names.  It is not the authoritative layout
        # consumed by SKEL.forward().  The actual pose vector is sliced
        # sequentially according to ``joints_dict[*].nb_dof`` and has 46
        # coordinates.
        self.raw_model_pose_names = [
            str(name) for name in self.model.pose_params_name
        ]
        self.pose_names = list(SKEL_POSE_NAMES)

        if self.joint_names != SKEL_BODY_NAMES:
            raise RuntimeError(
                "The installed SKEL model body order differs from the expected "
                f"v1.1.1 order.\nInstalled: {self.joint_names}\n"
                f"Expected:  {SKEL_BODY_NAMES}"
            )

        if int(self.model.num_q_params) != 46:
            raise RuntimeError(
                f"Expected SKEL num_q_params == 46, "
                f"got {self.model.num_q_params}."
            )

        installed_dof_counts: list[int] = []
        for joint in self.model.joints_dict:
            count = joint.nb_dof
            if hasattr(count, "item"):
                count = count.item()
            installed_dof_counts.append(int(count))

        expected_dof_counts = [
            len(SKEL_BODY_POSE_INDICES[body_name])
            for body_name in SKEL_BODY_NAMES
        ]

        if len(installed_dof_counts) != 24:
            raise RuntimeError(
                f"Expected 24 SKEL joint objects, "
                f"got {len(installed_dof_counts)}."
            )
        if sum(installed_dof_counts) != 46:
            raise RuntimeError(
                f"SKEL joints_dict defines {sum(installed_dof_counts)} "
                "pose coordinates instead of 46."
            )
        if installed_dof_counts != expected_dof_counts:
            raise RuntimeError(
                "The installed SKEL per-body DOF layout differs from the "
                "ForceBody-compatible layout.\n"
                f"Installed: {installed_dof_counts}\n"
                f"Expected:  {expected_dof_counts}"
            )

        print(
            "[SKEL] Ignoring pose_params_name metadata "
            f"({len(self.raw_model_pose_names)} entries); "
            "using the authoritative 46-DOF joints_dict layout."
        )

        model_parents = [-1] + [
            int(value) for value in self.model.parent.detach().cpu().tolist()
        ]
        if model_parents != SKEL_PARENT_INDICES:
            raise RuntimeError(
                "The installed SKEL parent tree differs from ForceBody's "
                "documented parent tree."
            )

        self.skin_weights = _as_dense_numpy(self.model.skin_weights)
        self.last_mapping: list[JointCoordinateMapping] = []
        self.last_translation_coupling: dict[str, np.ndarray] = {}
        self.last_zero_output: Any | None = None
        self.last_zero_root: np.ndarray | None = None

    @property
    def num_q_params(self) -> int:
        return int(self.model.num_q_params)

    def get_joints_verts(
        self,
        pose: torch.Tensor,
        th_betas: torch.Tensor | None = None,
        th_trans: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return SKEL skin vertices and anatomical joint positions."""
        pose = torch.as_tensor(pose, dtype=torch.float32)
        if pose.ndim == 1:
            pose = pose[None, :]
        if pose.shape[1] != 46:
            raise ValueError(f"Expected pose shape (B, 46), got {tuple(pose.shape)}.")

        batch_size = pose.shape[0]
        if th_betas is None:
            th_betas = torch.zeros((batch_size, 10), dtype=torch.float32)
        else:
            th_betas = torch.as_tensor(th_betas, dtype=torch.float32)
            if th_betas.ndim == 1:
                th_betas = th_betas[None, :]
            if th_betas.shape[0] == 1 and batch_size > 1:
                th_betas = th_betas.expand(batch_size, -1)

        if th_trans is None:
            th_trans = torch.zeros((batch_size, 3), dtype=torch.float32)
        else:
            th_trans = torch.as_tensor(th_trans, dtype=torch.float32)
            if th_trans.ndim == 1:
                th_trans = th_trans[None, :]

        with torch.no_grad():
            output = self.model.forward(
                poses=pose,
                betas=th_betas,
                trans=th_trans,
                skelmesh=False,
                pose_dep_bs=True,
            )
        return output.skin_verts, output.joints

    def _zero_forward(self, betas: torch.Tensor) -> Any:
        zero_pose = torch.zeros((1, 46), dtype=torch.float32)
        zero_trans = torch.zeros((1, 3), dtype=torch.float32)
        with torch.no_grad():
            return self.model.forward(
                poses=zero_pose,
                betas=betas,
                trans=zero_trans,
                skelmesh=False,
                pose_dep_bs=True,
            )

    def _infer_body_coordinates(
        self,
        betas: torch.Tensor,
        zero_output: Any,
        body_index: int,
    ) -> tuple[list[str], np.ndarray, np.ndarray, list[JointCoordinateMapping]]:
        """Infer effective zero-pose axes for one SKEL body.

        SKEL computes each local rotation as a chain wrapped by shape-dependent
        T-pose and A-pose frames.  A small finite difference of the official
        forward pass gives the effective axis in the parent frame without
        duplicating that internal frame logic.
        """
        body_name = self.joint_names[body_index]
        parent_index = SKEL_PARENT_INDICES[body_index]
        pose_indices = list(SKEL_BODY_POSE_INDICES[body_name])

        if parent_index < 0:
            return ["x", "y", "z"], np.eye(3), np.vstack(
                [pose_limit(self.pose_names[index]) for index in pose_indices]
            ), []

        zero_global_R = zero_output.joints_ori[0].detach().cpu().numpy()
        zero_global_p = zero_output.joints[0].detach().cpu().numpy()
        zero_local_R = _relative_rotation(
            zero_global_R, body_index, parent_index
        )
        zero_local_t = (
            zero_global_R[parent_index].T
            @ (zero_global_p[body_index] - zero_global_p[parent_index])
        )

        inferred: list[tuple[int, np.ndarray, np.ndarray]] = []
        for pose_index in pose_indices:
            perturbed_pose = torch.zeros((1, 46), dtype=torch.float32)
            perturbed_pose[0, pose_index] = self.axis_epsilon
            with torch.no_grad():
                perturbed = self.model.forward(
                    poses=perturbed_pose,
                    betas=betas,
                    trans=torch.zeros((1, 3), dtype=torch.float32),
                    skelmesh=False,
                    pose_dep_bs=True,
                )

            global_R = perturbed.joints_ori[0].detach().cpu().numpy()
            global_p = perturbed.joints[0].detach().cpu().numpy()
            local_R = _relative_rotation(global_R, body_index, parent_index)

            # SKEL left-multiplies the scalar rotations in q_to_rot().
            delta_R = local_R @ zero_local_R.T
            rotvec = Rotation.from_matrix(delta_R).as_rotvec()
            axis = rotvec / self.axis_epsilon
            norm = float(np.linalg.norm(axis))
            if not np.isfinite(norm) or norm < 1e-6:
                raise RuntimeError(
                    f"Could not infer a rotational axis for "
                    f"{body_name}:{self.pose_names[pose_index]}."
                )
            axis = axis / norm

            local_t = (
                global_R[parent_index].T
                @ (global_p[body_index] - global_p[parent_index])
            )
            translation_derivative = (
                local_t - zero_local_t
            ) / self.axis_epsilon
            inferred.append((pose_index, axis, translation_derivative))

        # MuJoCo applies multiple joints in XML order.  SKEL's CustomJoint
        # returns R_n ... R_1, so the default XML order is reversed.
        if self.reverse_multidof_order and len(inferred) > 1:
            inferred = list(reversed(inferred))

        channels = ["x", "y", "z"][: len(inferred)]
        axes = _complete_axis_matrix([item[1] for item in inferred])
        ranges = np.vstack(
            [pose_limit(self.pose_names[item[0]]) for item in inferred]
        )

        mappings = []
        for channel, (pose_index, axis, translation_derivative) in zip(
            channels, inferred
        ):
            mappings.append(
                JointCoordinateMapping(
                    body_name=body_name,
                    pose_index=pose_index,
                    pose_name=self.pose_names[pose_index],
                    channel=channel,
                    mjcf_joint_name=f"{body_name}_{channel}",
                    axis_parent_frame=axis.copy(),
                    translation_derivative=translation_derivative.copy(),
                )
            )
        return channels, axes, ranges, mappings

    def get_mesh_offsets(
        self,
        zero_pose: torch.Tensor | None = None,
        betas: torch.Tensor | np.ndarray | None = None,
        flatfoot: bool = False,
    ):
        """Return the same tuple contract as SMPL_Parser.get_mesh_offsets()."""
        if zero_pose is not None:
            zero_pose = torch.as_tensor(zero_pose, dtype=torch.float32)
            if torch.max(torch.abs(zero_pose)).item() > 1e-8:
                raise ValueError(
                    "SKELSim currently builds geometry only from the zero pose."
                )

        if betas is None:
            betas_tensor = torch.zeros((1, 10), dtype=torch.float32)
        else:
            betas_tensor = torch.as_tensor(betas, dtype=torch.float32)
            if betas_tensor.ndim == 1:
                betas_tensor = betas_tensor[None, :]
        if tuple(betas_tensor.shape) != (1, 10):
            raise ValueError(
                f"Expected one subject beta vector with shape (1, 10), "
                f"got {tuple(betas_tensor.shape)}."
            )

        zero_output = self._zero_forward(betas_tensor)
        self.last_zero_output = zero_output

        vertices = zero_output.skin_verts[0].detach().cpu().numpy().copy()
        joints = zero_output.joints_tpose[0].detach().cpu().numpy().copy()

        # Make the pelvis the XML model origin.  This makes the free-root qpos
        # and the ForceBody root trajectory substantially easier to interpret.
        zero_root = joints[0].copy()
        self.last_zero_root = zero_root
        vertices -= zero_root
        joints -= zero_root

        if flatfoot:
            minimum_y = float(np.min(vertices[:, 1]))
            foot_mask = vertices[:, 1] < minimum_y + 0.01
            vertices[foot_mask, 1] = float(np.mean(vertices[foot_mask, 1]))

        joint_offsets: dict[str, np.ndarray] = {}
        joint_parents: dict[str, str | None] = {}
        joint_axes: dict[str, np.ndarray] = {}
        joint_dofs: dict[str, list[str]] = {}
        joint_range: dict[str, np.ndarray] = {}
        mapping: list[JointCoordinateMapping] = []

        for body_index, body_name in enumerate(self.joint_names):
            parent_index = SKEL_PARENT_INDICES[body_index]
            if parent_index < 0:
                joint_offsets[body_name] = np.zeros(3, dtype=np.float64)
                joint_parents[body_name] = None
            else:
                joint_offsets[body_name] = (
                    joints[body_index] - joints[parent_index]
                )
                joint_parents[body_name] = self.joint_names[parent_index]

            channels, axes, ranges, body_mapping = (
                self._infer_body_coordinates(
                    betas_tensor, zero_output, body_index
                )
            )
            joint_dofs[body_name] = channels
            joint_axes[body_name] = axes
            joint_range[body_name] = ranges
            mapping.extend(body_mapping)

        self.last_mapping = mapping
        self.last_translation_coupling = {
            item.mjcf_joint_name: item.translation_derivative.copy()
            for item in mapping
        }

        internal_pose_indices = sorted(item.pose_index for item in mapping)
        if internal_pose_indices != list(range(3, 46)):
            raise RuntimeError(
                "Expected exactly 43 internal coordinates covering pose "
                f"indices 3..45; got {internal_pose_indices}."
            )

        contype = {1: list(self.joint_names)}
        conaffinity = {1: list(self.joint_names)}

        return (
            vertices,
            joints,
            self.skin_weights.copy(),
            list(self.joint_names),
            joint_offsets,
            joint_parents,
            joint_axes,
            joint_dofs,
            joint_range,
            contype,
            conaffinity,
        )
