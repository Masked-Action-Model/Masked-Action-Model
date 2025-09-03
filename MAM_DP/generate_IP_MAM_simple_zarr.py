import os
import numpy as np
import zarr

def generate_IP_MAM_simple_zarr():
    """
    生成包含MAMDataset所需数据结构的zarr文件
    - state: shape [num_episodes, 7], type: low_dim (前2列用作agent_pos)
    - action: shape [num_episodes, length, 8], type: low_dim
    - MAS: shape [num_episodes, length, 8], type: low_dim
    - img1: shape [num_episodes, 3, 512, 512], type: rgb (每个episode一张图片)
    - img2: shape [num_episodes, 3, 512, 512], type: rgb (每个episode一张图片)
    """
    
    # 设置输出路径
    output_path = "/home/ybli/dp/diffusion_policy/data/custom_IP_MAM_replay.zarr"
    
    # 确保目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 创建zarr文件
    store = zarr.DirectoryStore(output_path)
    root = zarr.open_group(store=store, mode='w')
    
    # 创建data组
    data_group = root.create_group('data')
    
    # 设定 episode 数量
    num_episodes = 1000
    steps_per_episode = 1  # 每个episode只有一个step
    
    # 1. state: shape [num_episodes, 7], type: low_dim (前2列是agent_pos)
    state_data = np.random.uniform(-1.0, 1.0, size=(num_episodes, 7)).astype(np.float32)
    data_group.create_dataset('state', data=state_data, chunks=(100, 7))
    
    # 为所有episode生成统一长度的MAS和action
    # 使用固定长度以便创建统一的数组结构
    max_length = 100  # 使用最大可能长度作为统一长度
    
    # 2. MAS: shape [num_episodes, max_length, 8]
    mas_data = np.full((num_episodes, max_length, 8), -999.0, dtype=np.float32)  # 用-999填充
    
    # 3. action: shape [num_episodes, max_length, 8] 
    action_data = np.full((num_episodes, max_length, 8), -999.0, dtype=np.float32)  # 用-999填充
    
    # 为每个episode生成实际数据
    for episode in range(num_episodes):
        # 为每个episode随机生成长度[50,100]
        episode_length = np.random.randint(50, 101)
        
        # 生成MAS: shape [episode_length, 8]
        mas_episode = np.random.uniform(-2.0, 2.0, size=(episode_length, 8)).astype(np.float32)
        
        # 50%的值设为-1
        mask = np.random.random(size=(episode_length, 8)) < 0.5
        mas_episode[mask] = -1.0
        
        # 生成action: 与MAS形状一致的随机值 [episode_length, 8]
        action_episode = np.random.uniform(-1.0, 1.0, size=(episode_length, 8)).astype(np.float32)
        
        # 填充到统一数组中
        mas_data[episode, :episode_length, :] = mas_episode
        action_data[episode, :episode_length, :] = action_episode
    
    # 创建MAS和action数据集，chunks设为100
    data_group.create_dataset('MAS', data=mas_data, chunks=(100, max_length, 8))
    data_group.create_dataset('action', data=action_data, chunks=(100, max_length, 8))
    
    # 4. img1: shape [num_episodes, 3, 512, 512], type: rgb (每个episode一张图片)
    img1_data = np.random.randint(0, 256, size=(num_episodes, 3, 512, 512), dtype=np.uint8).astype(np.float32) / 255.0
    data_group.create_dataset('img1', data=img1_data, chunks=(100, 3, 512, 512))
    
    # 5. img2: shape [num_episodes, 3, 512, 512], type: rgb (每个episode一张图片)
    img2_data = np.random.randint(0, 256, size=(num_episodes, 3, 512, 512), dtype=np.uint8).astype(np.float32) / 255.0
    data_group.create_dataset('img2', data=img2_data, chunks=(100, 3, 512, 512))
    
    # 创建meta组
    meta_group = root.create_group('meta')
    
    # 创建episode_ends（1000个episode，每个只有1个step）
    episode_ends = np.cumsum(np.full(shape=(num_episodes,), fill_value=steps_per_episode, dtype=np.int64))
    meta_group.create_dataset('episode_ends', data=episode_ends, chunks=(100,))
    
    print(f"生成zarr文件: {output_path}")
    print(f"Episode数量: {num_episodes}")
    print(f"每个episode的steps: {steps_per_episode}")
    print(f"  state: {state_data.shape} {state_data.dtype}")
    print(f"  MAS: {mas_data.shape} {mas_data.dtype}")
    print(f"  action: {action_data.shape} {action_data.dtype}")
    print(f"  img1: {img1_data.shape} {img1_data.dtype}")
    print(f"  img2: {img2_data.shape} {img2_data.dtype}")
    print(f"  episode_ends: {episode_ends.shape} {episode_ends.dtype} -> {episode_ends}")
    
    print("✓ 每个episode只有一个step")
    print("✓ state: [num_episodes, 7]")
    print("✓ MAS和action: [num_episodes, length, 8]")
    print("✓ img1, img2: [num_episodes, 3, 512, 512]")
    print("✓ 包含MAMDataset所需的所有键: ['img1', 'img2', 'state', 'MAS', 'action']")
    
    # 打印每个episode的实际数据长度信息
    for episode in range(min(10, num_episodes)):  # 只打印前10个episode的信息
        # 找到非填充值的实际长度
        valid_mask = mas_data[episode, :, 0] != -999.0
        actual_length = np.sum(valid_mask)
        print(f"  Episode {episode}: 实际数据长度 = {actual_length}")
    
    return output_path

if __name__ == "__main__":
    generate_IP_MAM_simple_zarr()
