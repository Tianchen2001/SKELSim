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

    def _compute_joint_frame_to_tpose(
        self,
        zero_output: Any,
    ) -> np.ndarray:
        """Return the static frame conjugation used by SKEL for each body.

        SKEL constructs the local rotation of body ``i`` as

            R_local = Rk01 @ Ra.T @ Rp(q) @ Ra @ Rk01.T

        where ``Rp`` is produced by the biomechanical joint object.  Defining

            C = Rk01 @ Ra.T

        gives ``R_local = C @ Rp @ C.T``.  Thus ``C`` maps a joint axis from
        the BSM/OpenSim parameter frame into the T-pose frame used by the
        generated MuJoCo bodies.

        We deliberately compute this from the same public SKEL routines as
        ``SKEL.forward()`` instead of trying to recover it from
        ``output.joints_ori``.  ``joints_ori`` contains global-chain and
        bone-mesh alignment rotations and is not the local joint transform.
        """
        joints_tpose = zero_output.joints_tpose.detach().clone()

        # This reproduces SKEL.forward() lines that form the parent-relative
        # T-pose joint vectors before compute_bone_orientation().
        joints_relative = joints_tpose.clone()
        joints_relative[:, 1:, :] = (
            joints_tpose[:, 1:, :]
            - joints_tpose[:, self.model.parent, :]
        )

        with torch.no_grad():
            rk01 = self.model.compute_bone_orientation(
                joints_tpose,
                joints_relative,
            )

        ra = self.model.apose_rel_transfo[:, :3, :3]
        ra = ra.to(device=rk01.device, dtype=rk01.dtype)
        ra = ra.unsqueeze(0).expand(rk01.shape[0], -1, -1, -1)

        joint_frame_to_tpose = torch.matmul(
            rk01,
            ra.transpose(-1, -2),
        )

        return (
            joint_frame_to_tpose[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )

    def _joint_object_rotation(
        self,
        body_index: int,
        local_dof_index: int,
        angle: float,
    ) -> np.ndarray:
        """Evaluate one SKEL joint object's ``Rp(q)`` matrix.

        Only the selected scalar coordinate is nonzero.  This preserves the
        exact sign conventions used by CustomJoint, WalkerKnee, PinJoint,
        ConstantCurvatureJoint, and EllipsoidJoint.
        """
        joint_object = self.model.joints_dict[body_index]

        nb_dof = joint_object.nb_dof
        if hasattr(nb_dof, "item"):
            nb_dof = nb_dof.item()
        nb_dof = int(nb_dof)

        if not 0 <= local_dof_index < nb_dof:
            raise IndexError(
                f"Body {self.joint_names[body_index]!r} has {nb_dof} DOFs, "
                f"but local DOF {local_dof_index} was requested."
            )

        device = self.model.skin_template_v.device
        dtype = self.model.skin_template_v.dtype
        q = torch.zeros((1, nb_dof), dtype=dtype, device=device)
        q[0, local_dof_index] = float(angle)

        with torch.no_grad():
            rp = joint_object.q_to_rot(q)

        return (
            rp[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )

    def _infer_body_coordinates(
        self,
        betas: torch.Tensor,
        zero_output: Any,
        joint_frame_to_tpose: np.ndarray,
        body_index: int,
    ) -> tuple[list[str], np.ndarray, np.ndarray, list[JointCoordinateMapping]]:
        """Infer MuJoCo hinge axes from SKEL's actual local joint transform.

        The generated MuJoCo bodies have no rest-pose ``quat``; at zero pose
        their frames are aligned with the SKEL T-pose coordinates.  Therefore
        the hinge axis written to MJCF must be expressed in that T-pose body
        frame.

        For each scalar coordinate we evaluate the corresponding SKEL joint
        object at +epsilon and -epsilon, conjugate its ``Rp`` matrix by the
        exact static SKEL frame ``C = Rk01 @ Ra.T``, and extract the signed
        infinitesimal axis from the central relative rotation.
        """
        body_name = self.joint_names[body_index]
        parent_index = SKEL_PARENT_INDICES[body_index]
        pose_indices = list(SKEL_BODY_POSE_INDICES[body_name])

        if parent_index < 0:
            return (
                ["x", "y", "z"],
                np.eye(3, dtype=np.float64),
                np.vstack(
                    [
                        pose_limit(self.pose_names[index])
                        for index in pose_indices
                    ]
                ),
                [],
            )

        c_matrix = np.asarray(
            joint_frame_to_tpose[body_index],
            dtype=np.float64,
        )
        identity_error = np.linalg.norm(
            c_matrix @ c_matrix.T - np.eye(3),
            ord="fro",
        )
        if not np.isfinite(identity_error) or identity_error > 1e-4:
            raise RuntimeError(
                f"Invalid SKEL frame matrix for {body_name}: "
                f"orthogonality error {identity_error:.3e}."
            )

        zero_global_p = (
            zero_output.joints[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        zero_local_t = (
            zero_global_p[body_index]
            - zero_global_p[parent_index]
        )

        inferred: list[tuple[int, np.ndarray, np.ndarray]] = []

        for local_dof_index, pose_index in enumerate(pose_indices):
            rp_plus = self._joint_object_rotation(
                body_index,
                local_dof_index,
                +self.axis_epsilon,
            )
            rp_minus = self._joint_object_rotation(
                body_index,
                local_dof_index,
                -self.axis_epsilon,
            )

            # Exact local SKEL rotation in the generated body's T-pose frame:
            # R_local(q) = C @ Rp(q) @ C.T
            local_r_plus = c_matrix @ rp_plus @ c_matrix.T
            local_r_minus = c_matrix @ rp_minus @ c_matrix.T

            # Central finite difference on SO(3).  For a one-parameter
            # rotation this is approximately Exp(2 * eps * axis).
            central_delta_r = local_r_plus @ local_r_minus.T
            rotvec = Rotation.from_matrix(
                central_delta_r
            ).as_rotvec()
            axis = rotvec / (2.0 * self.axis_epsilon)

            norm = float(np.linalg.norm(axis))
            if not np.isfinite(norm) or norm < 1e-6:
                raise RuntimeError(
                    f"Could not infer a rotational axis for "
                    f"{body_name}:{self.pose_names[pose_index]}."
                )
            axis = axis / norm

            # Translation coupling is separate from rotation.  It is nonzero
            # mainly for scapula and constant-curvature spine joints.  With
            # all ancestors at zero pose, the parent frame is aligned with
            # the T-pose/world frame, so the parent-child position difference
            # can be differentiated directly.
            perturbed_pose = torch.zeros(
                (1, 46),
                dtype=betas.dtype,
                device=betas.device,
            )
            perturbed_pose[0, pose_index] = self.axis_epsilon

            with torch.no_grad():
                perturbed = self.model.forward(
                    poses=perturbed_pose,
                    betas=betas,
                    trans=torch.zeros(
                        (1, 3),
                        dtype=betas.dtype,
                        device=betas.device,
                    ),
                    skelmesh=False,
                    pose_dep_bs=True,
                )

            perturbed_global_p = (
                perturbed.joints[0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            perturbed_local_t = (
                perturbed_global_p[body_index]
                - perturbed_global_p[parent_index]
            )
            translation_derivative = (
                perturbed_local_t - zero_local_t
            ) / self.axis_epsilon

            inferred.append(
                (
                    pose_index,
                    axis,
                    translation_derivative,
                )
            )

        # SKEL's CustomJoint composes scalar rotations as R_n ... R_1.
        # The existing SMPLSim writer serializes MuJoCo joints in channel
        # order, so reverse the scalar order by default.  This does not affect
        # one-DOF joints such as the knee.
        if self.reverse_multidof_order and len(inferred) > 1:
            inferred = list(reversed(inferred))

        channels = ["x", "y", "z"][: len(inferred)]
        axes = _complete_axis_matrix(
            [item[1] for item in inferred]
        )
        ranges = np.vstack(
            [
                pose_limit(self.pose_names[item[0]])
                for item in inferred
            ]
        )

        mappings: list[JointCoordinateMapping] = []
        for channel, (
            pose_index,
            axis,
            translation_derivative,
        ) in zip(channels, inferred):
            mappings.append(
                JointCoordinateMapping(
                    body_name=body_name,
                    pose_index=pose_index,
                    pose_name=self.pose_names[pose_index],
                    channel=channel,
                    mjcf_joint_name=f"{body_name}_{channel}",
                    axis_parent_frame=axis.copy(),
                    translation_derivative=(
                        translation_derivative.copy()
                    ),
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

        joint_frame_to_tpose = self._compute_joint_frame_to_tpose(
            zero_output
        )

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
                    betas_tensor,
                    zero_output,
                    joint_frame_to_tpose,
                    body_index,
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
