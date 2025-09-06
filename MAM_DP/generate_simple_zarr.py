import os
import numpy as np
import zarr

def generate_simple_zarr():
    """
    生成包含MAMDataset所需数据结构的zarr文件
    - state: shape [7], type: low_dim (前2列用作agent_pos)
    - action: shape [7], type: low_dim
    - condition: shape [100, 8], type: low_dim  
    - img1: shape [3, 512, 512], type: rgb (第一张图片)
    - img2: shape [3, 512, 512], type: rgb (第二张图片)
    确保所有数据字段的帧数一致
    """
    
    # 设置输出路径
    output_path = "/home/ybli/dp/diffusion_policy/data/custom_replay.zarr"
    
    # 确保目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 创建zarr文件
    store = zarr.DirectoryStore(output_path)
    root = zarr.open_group(store=store, mode='w')
    
    # 创建data组
    data_group = root.create_group('data')
    
    # 设定 episode 数量与每个 episode 的步数
    num_episodes = 10
    steps_per_episode = 100
    total_steps = num_episodes * steps_per_episode
    
    # 1. state: shape [total_steps, 7], type: low_dim (前2列是agent_pos)
    state_data = np.random.uniform(-1.0, 1.0, size=(total_steps, 7)).astype(np.float32)
    data_group.create_dataset('state', data=state_data, chunks=(100, 7))
    
    # 2. action: shape [total_steps, 7], type: low_dim
    action_data = np.random.uniform(-0.5, 0.5, size=(total_steps, 7)).astype(np.float32)
    data_group.create_dataset('action', data=action_data, chunks=(100, 7))
    
    # 3. condition: shape [total_steps, 100, 8], type: low_dim
    condition_data = np.random.uniform(-2.0, 2.0, size=(total_steps, 100, 8)).astype(np.float32)
    data_group.create_dataset('condition', data=condition_data, chunks=(100, 100, 8))
    
    # 4. img1: shape [total_steps, 3, 512, 512], type: rgb (第一张图片)
    img1_data = np.random.randint(0, 256, size=(total_steps, 3, 512, 512), dtype=np.uint8).astype(np.float32) / 255.0
    data_group.create_dataset('img1', data=img1_data, chunks=(100, 3, 512, 512))
    
    # 5. img2: shape [total_steps, 3, 512, 512], type: rgb (第二张图片)
    img2_data = np.random.randint(0, 256, size=(total_steps, 3, 512, 512), dtype=np.uint8).astype(np.float32) / 255.0
    data_group.create_dataset('img2', data=img2_data, chunks=(100, 3, 512, 512))
    
    # 创建meta组
    meta_group = root.create_group('meta')
    
    # 创建episode_ends（10个episode，每个长度为 steps_per_episode）
    episode_ends = np.cumsum(np.full(shape=(num_episodes,), fill_value=steps_per_episode, dtype=np.int64))
    meta_group.create_dataset('episode_ends', data=episode_ends, chunks=(num_episodes,))
    
    print(f"生成zarr文件: {output_path}")
    print(f"总步数 total_steps: {total_steps} = {num_episodes} * {steps_per_episode}")
    print(f"  state: {state_data.shape} {state_data.dtype}")
    print(f"  action: {action_data.shape} {action_data.dtype}")
    print(f"  condition: {condition_data.shape} {condition_data.dtype}")
    print(f"  img1: {img1_data.shape} {img1_data.dtype}")
    print(f"  img2: {img2_data.shape} {img2_data.dtype}")
    print(f"  episode_ends: {episode_ends.shape} {episode_ends.dtype} -> {episode_ends}")
    
    # 验证帧数一致性
    assert state_data.shape[0] == action_data.shape[0] == condition_data.shape[0] == img1_data.shape[0] == img2_data.shape[0], "帧数不一致！"
    print("✓ 帧数一致性验证通过")
    print("✓ 包含MAMDataset所需的所有键: ['img1', 'img2', 'state', 'action', 'condition']")
    
    return output_path

if __name__ == "__main__":
    generate_simple_zarr()