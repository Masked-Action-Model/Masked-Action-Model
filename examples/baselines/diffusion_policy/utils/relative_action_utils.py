from __future__ import annotations

import torch

from mani_skill.utils.geometry.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_euler_angles,
    matrix_to_quaternion,
    quaternion_invert,
    quaternion_multiply,
    quaternion_to_matrix,
)


PICKCUBE_QPOS_DIM = 9
PICKCUBE_QVEL_DIM = 9
PICKCUBE_IS_GRASPED_DIM = 1
PICKCUBE_TCP_POSE_DIM = 7
PICKCUBE_PANDA_ROOT_POS = (-0.615, 0.0, 0.0)


def _as_tensor_like(value, ref: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(value, device=ref.device, dtype=ref.dtype)


def _broadcast_ref_pose(ref_pose: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    ref_pose = ref_pose.to(device=target.device, dtype=target.dtype)
    while ref_pose.ndim < target.ndim:
        ref_pose = ref_pose.unsqueeze(-2)
    return torch.broadcast_to(ref_pose, target.shape[:-1] + (6,))


def _euler_to_quat(euler_xyz: torch.Tensor) -> torch.Tensor:
    return matrix_to_quaternion(euler_angles_to_matrix(euler_xyz, "XYZ"))


def _quat_to_euler(quat: torch.Tensor) -> torch.Tensor:
    return matrix_to_euler_angles(quaternion_to_matrix(quat), "XYZ")


def denormalize_abs_action(
    action_norm: torch.Tensor,
    action_min: torch.Tensor,
    action_max: torch.Tensor,
) -> torch.Tensor:
    action_abs = action_norm.clone()
    d = min(int(action_min.shape[0]), action_abs.shape[-1] - 1)
    mins = action_min[:d].to(device=action_abs.device, dtype=action_abs.dtype)
    maxs = action_max[:d].to(device=action_abs.device, dtype=action_abs.dtype)
    action_abs[..., :d] = mins + 0.5 * (action_abs[..., :d] + 1.0) * (maxs - mins)
    return action_abs


def normalize_abs_action(
    action_abs: torch.Tensor,
    action_min: torch.Tensor,
    action_max: torch.Tensor,
) -> torch.Tensor:
    action_norm = action_abs.clone()
    d = min(int(action_min.shape[0]), action_norm.shape[-1] - 1)
    mins = action_min[:d].to(device=action_norm.device, dtype=action_norm.dtype)
    maxs = action_max[:d].to(device=action_norm.device, dtype=action_norm.dtype)
    scale = torch.where(maxs > mins, maxs - mins, torch.ones_like(maxs))
    action_norm[..., :d] = ((action_norm[..., :d] - mins) / scale) * 2.0 - 1.0
    return action_norm


def tcp_pose_to_abs_euler(tcp_pose: torch.Tensor) -> torch.Tensor:
    tcp_pose = torch.as_tensor(tcp_pose)
    if tcp_pose.shape[-1] != 7:
        raise ValueError(f"tcp_pose last dim must be 7, got {tuple(tcp_pose.shape)}")
    return torch.cat([tcp_pose[..., :3], _quat_to_euler(tcp_pose[..., 3:7])], dim=-1)


def tcp_pose_to_panda_base_abs_euler(tcp_pose: torch.Tensor) -> torch.Tensor:
    """Convert PickCube obs tcp_pose from world frame to Panda EE controller frame."""
    abs_pose = tcp_pose_to_abs_euler(tcp_pose)
    root_pos = _as_tensor_like(PICKCUBE_PANDA_ROOT_POS, abs_pose)
    abs_pose = abs_pose.clone()
    abs_pose[..., :3] = abs_pose[..., :3] - root_pos
    return abs_pose


def extract_pickcube_tcp_pose_from_state(state: torch.Tensor) -> torch.Tensor:
    tcp_start = PICKCUBE_QPOS_DIM + PICKCUBE_QVEL_DIM + PICKCUBE_IS_GRASPED_DIM
    tcp_end = tcp_start + PICKCUBE_TCP_POSE_DIM
    if state.shape[-1] < tcp_end:
        raise ValueError(
            f"PickCube state dim must be at least {tcp_end}, got {tuple(state.shape)}"
        )
    return state[..., tcp_start:tcp_end]


def absolute_pose_to_relative(abs_pose: torch.Tensor, ref_pose: torch.Tensor) -> torch.Tensor:
    if abs_pose.shape[-1] != 6:
        raise ValueError(f"abs_pose last dim must be 6, got {tuple(abs_pose.shape)}")
    ref_pose = _broadcast_ref_pose(ref_pose, abs_pose)
    rel_pos = abs_pose[..., :3] - ref_pose[..., :3]
    q_target = _euler_to_quat(abs_pose[..., 3:6])
    q_ref = _euler_to_quat(ref_pose[..., 3:6])
    q_rel = quaternion_multiply(q_target, quaternion_invert(q_ref))
    return torch.cat([rel_pos, _quat_to_euler(q_rel)], dim=-1)


def relative_pose_to_absolute(relative_pose: torch.Tensor, ref_pose: torch.Tensor) -> torch.Tensor:
    if relative_pose.shape[-1] != 6:
        raise ValueError(
            f"relative_pose last dim must be 6, got {tuple(relative_pose.shape)}"
        )
    ref_pose = _broadcast_ref_pose(ref_pose, relative_pose)
    abs_pos = ref_pose[..., :3] + relative_pose[..., :3]
    q_rel = _euler_to_quat(relative_pose[..., 3:6])
    q_ref = _euler_to_quat(ref_pose[..., 3:6])
    q_abs = quaternion_multiply(q_rel, q_ref)
    return torch.cat([abs_pos, _quat_to_euler(q_abs)], dim=-1)


def absolute_action_sequence_to_relative(
    abs_seq: torch.Tensor,
    ref_pose: torch.Tensor,
) -> torch.Tensor:
    if abs_seq.shape[-1] != 7:
        raise ValueError(f"abs_seq last dim must be 7, got {tuple(abs_seq.shape)}")
    rel_pose = absolute_pose_to_relative(abs_seq[..., :6], ref_pose)
    return torch.cat([rel_pose, abs_seq[..., 6:7]], dim=-1)


def relative_action_sequence_to_absolute(
    relative_seq: torch.Tensor,
    ref_pose: torch.Tensor,
) -> torch.Tensor:
    if relative_seq.shape[-1] != 7:
        raise ValueError(
            f"relative_seq last dim must be 7, got {tuple(relative_seq.shape)}"
        )
    abs_pose = relative_pose_to_absolute(relative_seq[..., :6], ref_pose)
    return torch.cat([abs_pose, relative_seq[..., 6:7]], dim=-1)


def normalize_relative_action(
    relative_seq: torch.Tensor,
    pos_scale: float,
    rot_scale: float,
    clip: bool = True,
) -> torch.Tensor:
    out = relative_seq.clone()
    out[..., :3] = out[..., :3] / float(pos_scale)
    out[..., 3:6] = out[..., 3:6] / float(rot_scale)
    if clip:
        out[..., :6] = torch.clamp(out[..., :6], -1.0, 1.0)
    return out


def denormalize_relative_action(
    relative_norm_seq: torch.Tensor,
    pos_scale: float,
    rot_scale: float,
) -> torch.Tensor:
    out = relative_norm_seq.clone()
    out[..., :3] = out[..., :3] * float(pos_scale)
    out[..., 3:6] = out[..., 3:6] * float(rot_scale)
    return out


def relative_sequence_to_delta_actions(relative_seq: torch.Tensor) -> torch.Tensor:
    if relative_seq.shape[-1] != 7:
        raise ValueError(
            f"relative_seq last dim must be 7, got {tuple(relative_seq.shape)}"
        )
    delta = relative_seq.clone()
    if relative_seq.shape[-2] <= 1:
        return delta
    prev = relative_seq[..., :-1, :]
    curr = relative_seq[..., 1:, :]
    delta[..., 1:, :3] = curr[..., :3] - prev[..., :3]
    q_curr = _euler_to_quat(curr[..., 3:6])
    q_prev = _euler_to_quat(prev[..., 3:6])
    q_delta = quaternion_multiply(q_curr, quaternion_invert(q_prev))
    delta[..., 1:, 3:6] = _quat_to_euler(q_delta)
    delta[..., 1:, 6] = curr[..., 6]
    return delta


def normalize_delta_for_pd_ee_delta_pose(
    delta_seq: torch.Tensor,
    pos_scale: float = 0.1,
    rot_scale: float = 0.1,
    clip: bool = True,
) -> torch.Tensor:
    out = delta_seq.clone()
    out[..., :3] = out[..., :3] / float(pos_scale)
    # PDEEPoseController multiplies normalized rotation by rot_lower, which is
    # negative for the default Panda delta-pose controller.
    out[..., 3:6] = out[..., 3:6] / float(-rot_scale)
    if clip:
        out[..., :3] = torch.clamp(out[..., :3], -1.0, 1.0)
        rot_norm = torch.linalg.norm(out[..., 3:6], dim=-1, keepdim=True)
        out[..., 3:6] = torch.where(
            rot_norm > 1.0,
            out[..., 3:6] / torch.clamp(rot_norm, min=1e-6),
            out[..., 3:6],
        )
    return out


def denormalize_pd_ee_delta_pose_action(
    delta_norm_seq: torch.Tensor,
    pos_scale: float = 0.1,
    rot_scale: float = 0.1,
) -> torch.Tensor:
    out = delta_norm_seq.clone()
    out[..., :3] = torch.clamp(out[..., :3], -1.0, 1.0) * float(pos_scale)
    rot = out[..., 3:6]
    rot_norm = torch.linalg.norm(rot, dim=-1, keepdim=True)
    rot = torch.where(rot_norm > 1.0, rot / torch.clamp(rot_norm, min=1e-6), rot)
    out[..., 3:6] = rot * float(-rot_scale)
    return out


def integrate_delta_actions(ref_pose: torch.Tensor, delta_seq: torch.Tensor) -> torch.Tensor:
    if delta_seq.shape[-1] != 7:
        raise ValueError(f"delta_seq last dim must be 7, got {tuple(delta_seq.shape)}")
    if delta_seq.shape[-2] == 0:
        return delta_seq.clone()

    positions = []
    quats = []
    current_pos = delta_seq[..., 0, :3]
    current_quat = _euler_to_quat(delta_seq[..., 0, 3:6])
    positions.append(current_pos)
    quats.append(current_quat)

    for step_idx in range(1, delta_seq.shape[-2]):
        current_pos = current_pos + delta_seq[..., step_idx, :3]
        delta_quat = _euler_to_quat(delta_seq[..., step_idx, 3:6])
        current_quat = quaternion_multiply(delta_quat, current_quat)
        positions.append(current_pos)
        quats.append(current_quat)

    rel_pose = torch.cat(
        [
            torch.stack(positions, dim=-2),
            _quat_to_euler(torch.stack(quats, dim=-2)),
        ],
        dim=-1,
    )
    relative_seq = torch.cat([rel_pose, delta_seq[..., 6:7]], dim=-1)
    return relative_action_sequence_to_absolute(relative_seq, ref_pose)


def convert_masked_mas_to_relative(
    mas_window: torch.Tensor,
    mas_mask: torch.Tensor,
    ref_pose: torch.Tensor,
    action_min: torch.Tensor,
    action_max: torch.Tensor,
    relative_pos_scale: float,
    relative_rot_scale: float,
    mask_value: float = 0.0,
) -> torch.Tensor:
    if mas_window.shape[-1] != 8:
        raise ValueError(f"mas_window last dim must be 8, got {tuple(mas_window.shape)}")
    if mas_mask.shape != mas_window.shape:
        raise ValueError(
            f"mas_mask shape must match mas_window, got {tuple(mas_mask.shape)} vs {tuple(mas_window.shape)}"
        )

    out = mas_window.clone()
    keep = mas_mask[..., :7] > 0.5
    abs_action = denormalize_abs_action(out[..., :7], action_min, action_max)
    ref_pose_b = _broadcast_ref_pose(ref_pose, abs_action[..., :6])
    filled_abs_pose = torch.where(keep[..., :6], abs_action[..., :6], ref_pose_b)
    rel_pose = absolute_pose_to_relative(filled_abs_pose, ref_pose_b)
    rel_action = torch.cat([rel_pose, abs_action[..., 6:7]], dim=-1)
    rel_action = normalize_relative_action(
        rel_action,
        pos_scale=relative_pos_scale,
        rot_scale=relative_rot_scale,
        clip=True,
    )
    out[..., :7] = torch.where(
        keep,
        rel_action,
        _as_tensor_like(mask_value, out),
    )
    out[..., 7] = mas_window[..., 7]
    return out
