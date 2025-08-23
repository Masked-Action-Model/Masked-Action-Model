import h5py
import numpy as np
import argparse
import os
import cv2
import zarr
from termcolor import cprint
import copy

def extract_frame_from_video(video_path, frame_idx):
    """
    从视频中提取指定帧的图片
    
    Args:
        video_path: 视频文件路径
        frame_idx: 帧索引
    
    Returns:
        frame: 提取的帧图片 (H, W, C)
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 检查帧索引是否有效
    if frame_idx < 0 or frame_idx >= total_frames:
        cap.release()
        raise ValueError(f"帧索引 {frame_idx} 超出范围 [0, {total_frames-1}]")
    
    # 设置帧位置
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    
    # 读取帧
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"无法读取第 {frame_idx} 帧")
    
    return frame

def validate_video_frames(h5_path, traj_key):
    """
    验证视频总帧数是否等于trajectory的长度
    
    Args:
        h5_path: .h5文件路径
        traj_key: trajectory的key (如 'tra_0')
    
    Returns:
        bool: 验证是否通过
    """
    # 获取trajectory长度
    with h5py.File(h5_path, 'r') as f:
        if 'trajectories' not in f or traj_key not in f['trajectories']:
            raise ValueError(f"在.h5文件中找不到 {traj_key}")
        
        traj_data = f['trajectories'][traj_key]
        if 'action' not in traj_data:
            raise ValueError(f"在 {traj_key} 中找不到 'action' 数据")
        
        traj_length = traj_data['action'].shape[0]
    
    # 检查两个视角的视频文件
    h5_dir = os.path.dirname(h5_path)
    view0_video = os.path.join(h5_dir, '0', f'{traj_key.split("_")[1]}.mp4')
    view1_video = os.path.join(h5_dir, '1', f'{traj_key.split("_")[1]}.mp4')
    
    # 检查视频文件是否存在
    if not os.path.exists(view0_video):
        raise FileNotFoundError(f"找不到视角0视频文件: {view0_video}")
    if not os.path.exists(view1_video):
        raise FileNotFoundError(f"找不到视角1视频文件: {view1_video}")
    
    # 检查视角0视频帧数
    cap0 = cv2.VideoCapture(view0_video)
    if not cap0.isOpened():
        cap0.release()
        raise ValueError(f"无法打开视角0视频文件: {view0_video}")
    
    view0_frames = int(cap0.get(cv2.CAP_PROP_FRAME_COUNT))
    cap0.release()
    
    # 检查视角1视频帧数
    cap1 = cv2.VideoCapture(view1_video)
    if not cap1.isOpened():
        cap1.release()
        raise ValueError(f"无法打开视角1视频文件: {view1_video}")
    
    view1_frames = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    cap1.release()
    
    # 验证帧数是否匹配
    if view0_frames != traj_length:
        raise ValueError(f"视角0视频帧数 ({view0_frames}) 与trajectory长度 ({traj_length}) 不匹配")
    
    if view1_frames != traj_length:
        raise ValueError(f"视角1视频帧数 ({view1_frames}) 与trajectory长度 ({traj_length}) 不匹配")
    
    cprint(f"✓ {traj_key}: 视频帧数验证通过 (长度: {traj_length})", 'green')
    return True

def extract_trajectory_frames(h5_path, traj_key):
    """
    提取trajectory对应的所有帧图片
    
    Args:
        h5_path: .h5文件路径
        traj_key: trajectory的key
    
    Returns:
        tuple: (view0_frames, view1_frames, traj_data)
    """
    h5_dir = os.path.dirname(h5_path)
    view0_video = os.path.join(h5_dir, '0', f'{traj_key.split("_")[1]}.mp4')
    view1_video = os.path.join(h5_dir, '1', f'{traj_key.split("_")[1]}.mp4')
    
    # 获取trajectory长度
    with h5py.File(h5_path, 'r') as f:
        traj_data = f['trajectories'][traj_key]
        traj_length = traj_data['action'].shape[0]
    
    # 提取两个视角的所有帧
    view0_frames = []
    view1_frames = []
    
    for frame_idx in range(traj_length):
        try:
            frame0 = extract_frame_from_video(view0_video, frame_idx)
            frame1 = extract_frame_from_video(view1_video, frame_idx)
            
            view0_frames.append(frame0)
            view1_frames.append(frame1)
            
        except Exception as e:
            raise RuntimeError(f"提取第 {frame_idx} 帧时出错: {str(e)}")
    
    return view0_frames, view1_frames, traj_data

def main(args):
    input_h5 = args.input
    output_zarr = args.output
    
    if not os.path.exists(input_h5):
        cprint(f"输入文件不存在: {input_h5}", 'red')
        return
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_zarr), exist_ok=True)
    
    if os.path.exists(output_zarr):
        cprint(f'数据已存在于 {output_zarr}', 'red')
        cprint("如果要覆盖，请先删除现有目录。", "red")
        cprint("是否要覆盖? (y/n)", "red")
        user_input = 'y'  # 默认覆盖
        if user_input == 'y':
            cprint(f'覆盖 {output_zarr}', 'red')
            import shutil
            shutil.rmtree(output_zarr)
        else:
            cprint('退出', 'red')
            return
    
    cprint(f"开始处理 {input_h5}", 'yellow')
    
    # 存储所有数据
    all_view0_frames = []
    all_view1_frames = []
    all_actions = []
    all_states = []
    all_full_states = []
    episode_ends = []
    
    total_count = 0
    
    try:
        with h5py.File(input_h5, 'r') as f:
            if 'trajectories' not in f:
                raise ValueError("在.h5文件中找不到 'trajectories' 组")
            
            trajectory_keys = list(f['trajectories'].keys())
            cprint(f"找到 {len(trajectory_keys)} 个trajectory", 'yellow')
            
            for traj_key in trajectory_keys:
                cprint(f"处理 {traj_key}...", 'cyan')
                
                try:
                    # 验证视频帧数
                    validate_video_frames(input_h5, traj_key)
                    
                    # 提取帧和trajectory数据
                    view0_frames, view1_frames, traj_data = extract_trajectory_frames(input_h5, traj_key)
                    
                    # 获取trajectory数据
                    actions = traj_data['action'][()]
                    states = traj_data['state'][()] if 'state' in traj_data else None
                    full_states = traj_data['full_state'][()] if 'full_state' in traj_data else None
                    
                    # 添加到总数据中
                    all_view0_frames.extend(view0_frames)
                    all_view1_frames.extend(view1_frames)
                    all_actions.extend([actions])
                    if states is not None:
                        all_states.extend([states])
                    if full_states is not None:
                        all_full_states.extend([full_states])
                    
                    # 更新episode结束位置
                    total_count += len(view0_frames)
                    episode_ends.append(total_count)
                    
                    cprint(f"✓ {traj_key} 处理完成，帧数: {len(view0_frames)}", 'green')
                    
                except Exception as e:
                    cprint(f"✗ 处理 {traj_key} 时出错: {str(e)}", 'red')
                    continue
        
        if not all_view0_frames:
            raise RuntimeError("没有成功处理任何trajectory")
        
        cprint(f"开始保存到zarr文件: {output_zarr}", 'yellow')
        
        # 创建zarr文件
        zarr_root = zarr.group(output_zarr)
        zarr_data = zarr_root.create_group('data')
        zarr_meta = zarr_root.create_group('meta')
        
        # 转换数据格式
        view0_arrays = np.stack(all_view0_frames, axis=0)
        view1_arrays = np.stack(all_view1_frames, axis=0)
        
        # 确保图像格式正确 (BGR -> RGB)
        view0_arrays = cv2.cvtColor(view0_arrays, cv2.COLOR_BGR2RGB)
        view1_arrays = cv2.cvtColor(view1_arrays, cv2.COLOR_BGR2RGB)
        
        # 转换action数据格式
        action_arrays = np.concatenate(all_actions, axis=0)
        
        # 设置压缩器
        compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
        
        # 设置chunk大小
        img_chunk_size = (100, view0_arrays.shape[1], view0_arrays.shape[2], view0_arrays.shape[3])
        action_chunk_size = (100, action_arrays.shape[1])
        
        # 保存数据
        zarr_data.create_dataset('view0', data=view0_arrays, chunks=img_chunk_size, 
                                dtype='uint8', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('view1', data=view1_arrays, chunks=img_chunk_size, 
                                dtype='uint8', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, 
                                dtype='float32', overwrite=True, compressor=compressor)
        
        # 保存episode信息
        episode_ends_arrays = np.array(episode_ends)
        zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, 
                                dtype='int64', overwrite=True, compressor=compressor)
        
        # 如果有其他数据，也保存
        if all_states:
            state_arrays = np.concatenate(all_states, axis=0)
            state_chunk_size = (100, state_arrays.shape[1])
            zarr_data.create_dataset('state', data=state_arrays, chunks=state_chunk_size, 
                                    dtype='float32', overwrite=True, compressor=compressor)
        
        if all_full_states:
            full_state_arrays = np.concatenate(all_full_states, axis=0)
            full_state_chunk_size = (100, full_state_arrays.shape[1])
            zarr_data.create_dataset('full_state', data=full_state_arrays, chunks=full_state_chunk_size, 
                                    dtype='float32', overwrite=True, compressor=compressor)
        
        cprint(f'-'*50, 'cyan')
        cprint(f'view0 shape: {view0_arrays.shape}, range: [{np.min(view0_arrays)}, {np.max(view0_arrays)}]', 'green')
        cprint(f'view1 shape: {view1_arrays.shape}, range: [{np.min(view1_arrays)}, {np.max(view1_arrays)}]', 'green')
        cprint(f'action shape: {action_arrays.shape}, range: [{np.min(action_arrays)}, {np.max(action_arrays)}]', 'green')
        cprint(f'episode_ends: {episode_ends_arrays}', 'green')
        cprint(f'成功保存zarr文件到 {output_zarr}', 'green')
        
    except Exception as e:
        cprint(f"处理过程中出错: {str(e)}", 'red')
        raise
    
    finally:
        # 清理内存
        if 'all_view0_frames' in locals():
            del all_view0_frames, all_view1_frames, all_actions, all_states, all_full_states
        if 'zarr_root' in locals():
            del zarr_root, zarr_data, zarr_meta

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default="/data1/tangjielong_2/mask_action_model/Masked-Action-Model/demos/PlugCharger-v1/motionplanning/20250716_192215_mask.h5",  help='输入原始.h5文件路径')
    parser.add_argument('--output', type=str, default="./output_zarr/test.zarr", help='输出.zarr文件路径')
    
    args = parser.parse_args()
    main(args) 