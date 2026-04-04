import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image

import mani_skill.envs
from mani_skill.utils.building import actors
from mani_skill.utils.structs.pose import Pose


ACTION_MIN = np.array(
    [
        0.5085554718971252,
        -0.10290590673685074,
        0.017135675996541977,
        -3.141592502593994,
        -0.1091873049736023,
        -0.797687828540802,
    ],
    dtype=np.float32,
)

ACTION_MAX = np.array(
    [
        0.7137041091918945,
        0.10304217785596848,
        0.3223087191581726,
        3.1415927410125732,
        0.10959156602621078,
        0.8309240937232971,
    ],
    dtype=np.float32,
)

PANDA_BASE_WORLD_POS = np.array([-0.615, 0.0, 0.0], dtype=np.float32)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize PickCube pd_ee_pose xyz min/max range in world coordinates."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("examples/baselines/diffusion_policy/pickcube_pd_ee_pose_world_range.png"),
        help="Output image path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Environment reset seed for a deterministic scene snapshot.",
    )
    parser.add_argument(
        "--edge-points",
        type=int,
        default=9,
        help="Number of small spheres sampled per box edge.",
    )
    parser.add_argument(
        "--edge-radius",
        type=float,
        default=0.006,
        help="Radius of small spheres on box edges.",
    )
    parser.add_argument(
        "--corner-radius",
        type=float,
        default=0.01,
        help="Radius of corner spheres.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Output image width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1280,
        help="Output image height.",
    )
    return parser.parse_args()


def base_xyz_to_world(xyz_base: np.ndarray) -> np.ndarray:
    # PickCube-v1 uses panda at [-0.615, 0, 0] with identity rotation.
    return xyz_base + PANDA_BASE_WORLD_POS


def bbox_corners(min_xyz: np.ndarray, max_xyz: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [min_xyz[0], min_xyz[1], min_xyz[2]],
            [min_xyz[0], min_xyz[1], max_xyz[2]],
            [min_xyz[0], max_xyz[1], min_xyz[2]],
            [min_xyz[0], max_xyz[1], max_xyz[2]],
            [max_xyz[0], min_xyz[1], min_xyz[2]],
            [max_xyz[0], min_xyz[1], max_xyz[2]],
            [max_xyz[0], max_xyz[1], min_xyz[2]],
            [max_xyz[0], max_xyz[1], max_xyz[2]],
        ],
        dtype=np.float32,
    )


def bbox_edge_points(corners: np.ndarray, points_per_edge: int) -> np.ndarray:
    edge_ids = [
        (0, 1),
        (0, 2),
        (0, 4),
        (1, 3),
        (1, 5),
        (2, 3),
        (2, 6),
        (3, 7),
        (4, 5),
        (4, 6),
        (5, 7),
        (6, 7),
    ]
    ts = np.linspace(0.0, 1.0, points_per_edge, dtype=np.float32)
    points = []
    for a, b in edge_ids:
        pa = corners[a]
        pb = corners[b]
        for t in ts:
            points.append((1.0 - t) * pa + t * pb)
    return np.unique(np.round(np.asarray(points, dtype=np.float32), 6), axis=0)


def spawn_spheres(env, xyz_points: np.ndarray, radius: float, color, name_prefix: str):
    for i, xyz in enumerate(xyz_points):
        pose = Pose.create_from_pq(xyz.tolist(), [1, 0, 0, 0])
        actors.build_sphere(
            env.scene,
            radius=radius,
            color=color,
            name=f"{name_prefix}_{i}",
            body_type="static",
            initial_pose=pose,
        )


def save_render(env, out_path: Path):
    rgb = env.render()
    if hasattr(rgb, "detach"):
        rgb = rgb.detach().cpu().numpy()
    rgb = np.asarray(rgb)
    if rgb.ndim == 4 and rgb.shape[0] == 1:
        rgb = rgb[0]
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(out_path)


def main():
    args = parse_args()

    min_xyz_world = base_xyz_to_world(ACTION_MIN[:3])
    max_xyz_world = base_xyz_to_world(ACTION_MAX[:3])
    corners = bbox_corners(min_xyz_world, max_xyz_world)
    edge_points = bbox_edge_points(corners, points_per_edge=args.edge_points)

    env = gym.make(
        "PickCube-v1",
        obs_mode="state",
        reward_mode="sparse",
        control_mode="pd_ee_pose",
        render_mode="rgb_array",
        max_episode_steps=100,
        human_render_camera_configs=dict(shader_pack="default", width=args.width, height=args.height),
    )
    env.reset(seed=args.seed)

    spawn_spheres(
        env,
        edge_points,
        radius=args.edge_radius,
        color=[1.0, 0.55, 0.15, 1.0],
        name_prefix="range_edge",
    )
    spawn_spheres(
        env,
        corners,
        radius=args.corner_radius,
        color=[0.9, 0.1, 0.1, 1.0],
        name_prefix="range_corner",
    )
    env.scene.update_render()

    save_render(env, args.out.resolve())
    print("Saved image to:", args.out.resolve())
    print("Base-frame xyz min:", ACTION_MIN[:3])
    print("Base-frame xyz max:", ACTION_MAX[:3])
    print("World-frame xyz min:", min_xyz_world)
    print("World-frame xyz max:", max_xyz_world)

    env.close()


if __name__ == "__main__":
    main()
