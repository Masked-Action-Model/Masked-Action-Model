#!/usr/bin/env python3
"""Generate end-effector action poses from recorded Franka joint trajectories."""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_DATASET = Path("pick_cup_place_next_to_bowl/formal")
DEFAULT_URDF = Path("/home/robot/franka_ros2_ws/src/franka_description/urdfs/fr3.urdf")
ARM_JOINT_NAMES = [f"fr3_joint{i}" for i in range(1, 8)]


@dataclass(frozen=True)
class Joint:
    name: str
    joint_type: str
    parent: str
    child: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray


def parse_vector(value: str | None, default: tuple[float, float, float]) -> np.ndarray:
    if not value:
        return np.asarray(default, dtype=float)
    return np.asarray([float(x) for x in value.split()], dtype=float)


def rotation_from_rpy(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def rotation_about_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        return np.eye(3)
    x, y, z = axis / norm
    c = math.cos(angle)
    s = math.sin(angle)
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=float,
    )


def transform_from_xyz_rpy(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation_from_rpy(rpy)
    transform[:3, 3] = xyz
    return transform


def joint_motion_transform(joint: Joint, value: float) -> np.ndarray:
    transform = transform_from_xyz_rpy(joint.xyz, joint.rpy)
    if joint.joint_type in ("revolute", "continuous"):
        motion = np.eye(4, dtype=float)
        motion[:3, :3] = rotation_about_axis(joint.axis, value)
        transform = transform @ motion
    elif joint.joint_type == "prismatic":
        motion = np.eye(4, dtype=float)
        motion[:3, 3] = joint.axis * value
        transform = transform @ motion
    return transform


def quat_from_rotation_matrix(rot: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rot))
    if trace > 0.0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (rot[2, 1] - rot[1, 2]) * s
        y = (rot[0, 2] - rot[2, 0]) * s
        z = (rot[1, 0] - rot[0, 1]) * s
    else:
        diag = np.diag(rot)
        idx = int(np.argmax(diag))
        if idx == 0:
            s = 2.0 * math.sqrt(max(0.0, 1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]))
            w = (rot[2, 1] - rot[1, 2]) / s
            x = 0.25 * s
            y = (rot[0, 1] + rot[1, 0]) / s
            z = (rot[0, 2] + rot[2, 0]) / s
        elif idx == 1:
            s = 2.0 * math.sqrt(max(0.0, 1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]))
            w = (rot[0, 2] - rot[2, 0]) / s
            x = (rot[0, 1] + rot[1, 0]) / s
            y = 0.25 * s
            z = (rot[1, 2] + rot[2, 1]) / s
        else:
            s = 2.0 * math.sqrt(max(0.0, 1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]))
            w = (rot[1, 0] - rot[0, 1]) / s
            x = (rot[0, 2] + rot[2, 0]) / s
            y = (rot[1, 2] + rot[2, 1]) / s
            z = 0.25 * s
    quat = np.asarray([x, y, z, w], dtype=float)
    return quat / np.linalg.norm(quat)


def parse_urdf_joints(urdf_path: Path) -> list[Joint]:
    root = ET.parse(urdf_path).getroot()
    joints: list[Joint] = []
    for node in root.findall("joint"):
        origin = node.find("origin")
        axis = node.find("axis")
        joints.append(
            Joint(
                name=node.attrib["name"],
                joint_type=node.attrib.get("type", "fixed"),
                parent=node.find("parent").attrib["link"],
                child=node.find("child").attrib["link"],
                xyz=parse_vector(origin.attrib.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0)),
                rpy=parse_vector(origin.attrib.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0)),
                axis=parse_vector(axis.attrib.get("xyz") if axis is not None else None, (0.0, 0.0, 1.0)),
            )
        )
    return joints


def add_default_franka_hand(joints: list[Joint]) -> list[Joint]:
    names = {joint.name for joint in joints}
    out = list(joints)
    if "fr3_hand_joint" not in names:
        out.append(
            Joint(
                name="fr3_hand_joint",
                joint_type="fixed",
                parent="fr3_link8",
                child="fr3_hand",
                xyz=np.asarray([0.0, 0.0, 0.0]),
                rpy=np.asarray([0.0, 0.0, -math.pi / 4.0]),
                axis=np.asarray([0.0, 0.0, 1.0]),
            )
        )
    if "fr3_hand_tcp_joint" not in names:
        out.append(
            Joint(
                name="fr3_hand_tcp_joint",
                joint_type="fixed",
                parent="fr3_hand",
                child="fr3_hand_tcp",
                xyz=np.asarray([0.0, 0.0, 0.1034]),
                rpy=np.asarray([0.0, 0.0, 0.0]),
                axis=np.asarray([0.0, 0.0, 1.0]),
            )
        )
    return out


def find_chain(joints: list[Joint], base_link: str, ee_link: str) -> list[Joint]:
    children: dict[str, list[Joint]] = {}
    for joint in joints:
        children.setdefault(joint.parent, []).append(joint)

    stack: list[tuple[str, list[Joint]]] = [(base_link, [])]
    seen = set()
    while stack:
        link, chain = stack.pop()
        if link == ee_link:
            return chain
        if link in seen:
            continue
        seen.add(link)
        for joint in children.get(link, []):
            stack.append((joint.child, [*chain, joint]))
    raise ValueError(f"No URDF chain from {base_link} to {ee_link}")


def fk_pose(chain: list[Joint], joint_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q_by_name = {name: float(joint_values[i]) for i, name in enumerate(ARM_JOINT_NAMES)}
    transform = np.eye(4, dtype=float)
    for joint in chain:
        transform = transform @ joint_motion_transform(joint, q_by_name.get(joint.name, 0.0))
    pose = np.zeros(7, dtype=float)
    pose[0:3] = transform[:3, 3]
    pose[3:7] = quat_from_rotation_matrix(transform[:3, :3])
    return pose, transform


def compute_fk(chain: list[Joint], joint_positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    poses = np.zeros((joint_positions.shape[0], 7), dtype=np.float64)
    matrices = np.zeros((joint_positions.shape[0], 4, 4), dtype=np.float64)
    for i, q in enumerate(joint_positions):
        poses[i], matrices[i] = fk_pose(chain, q)
        if i > 0 and float(np.dot(poses[i - 1, 3:7], poses[i, 3:7])) < 0.0:
            poses[i, 3:7] *= -1.0
    return poses, matrices


def natural_traj_key(path: Path) -> int:
    try:
        return int(path.name.split("_", 1)[1])
    except (IndexError, ValueError):
        return -1


def process_traj(traj_dir: Path, chain: list[Joint], args: argparse.Namespace) -> str:
    source = traj_dir / args.joint_phase / "joint_trajectory.npz"
    if not source.exists():
        return f"{traj_dir.name}: skip missing {source.relative_to(traj_dir)}"

    data = np.load(source)
    timestamps = np.asarray(data["timestamps"], dtype=np.float64)
    joint_positions = np.asarray(data["joint_positions"], dtype=np.float64)
    if joint_positions.ndim != 2 or joint_positions.shape[1] < 7:
        raise ValueError(f"{source}: joint_positions must have shape (N, >=7)")
    joint_positions = joint_positions[:, :7]

    poses, matrices = compute_fk(chain, joint_positions)
    out_path = traj_dir / "replay" / "action_FK.npz"
    if out_path.exists() and not args.overwrite:
        return f"{traj_dir.name}: exists, use --overwrite"

    if not args.dry_run:
        np.savez_compressed(
            out_path,
            timestamps=timestamps,
            action_FK=poses,
            ee_pose=poses,
            positions=poses[:, 0:3],
            quaternions=poses[:, 3:7],
            transforms=matrices,
            joint_positions=joint_positions,
            joint_names=np.asarray(ARM_JOINT_NAMES),
            source_joint_file=np.asarray(str(source)),
            source_joint_phase=np.asarray(args.joint_phase),
            urdf_path=np.asarray(str(args.urdf)),
            base_link=np.asarray(args.base_link),
            ee_link=np.asarray(args.ee_link),
        )
    return f"{traj_dir.name}: wrote replay/action_FK.npz ({poses.shape[0]} samples)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--base-link", default="fr3_link0")
    parser.add_argument("--ee-link", default="fr3_hand_tcp")
    parser.add_argument("--joint-phase", choices=("replay", "teach"), default="replay")
    parser.add_argument("--no-default-franka-hand", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    joints = parse_urdf_joints(args.urdf)
    if not args.no_default_franka_hand:
        joints = add_default_franka_hand(joints)
    chain = find_chain(joints, args.base_link, args.ee_link)
    movable = [joint.name for joint in chain if joint.name in ARM_JOINT_NAMES]
    if movable != ARM_JOINT_NAMES:
        raise ValueError(f"FK chain movable joints are {movable}, expected {ARM_JOINT_NAMES}")

    traj_dirs = sorted(
        [path for path in args.dataset.glob("traj_*") if path.is_dir()],
        key=natural_traj_key,
    )
    if not traj_dirs:
        raise SystemExit(f"No traj_* directories found under {args.dataset}")

    for traj_dir in traj_dirs:
        print(process_traj(traj_dir, chain, args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
