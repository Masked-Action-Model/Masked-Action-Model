import h5py
import numpy as np
import argparse
import os
import cv2
import zarr
from termcolor import cprint
import copy
import subprocess
import tempfile

def extract_frame_from_video_gpu(video_path, frame_idx, use_gpu=True):
    """
    使用GPU加速从视频中提取指定帧的图片
    
    Args:
        video_path: 视频文件路径
        frame_idx: 帧索引
        use_gpu: 是否使用GPU加速
    
    Returns:
        frame: 提取的帧图片 (H, W, C)
    """
    if use_gpu:
        try:
            # 使用FFmpeg GPU加速提取帧
            return extract_frame_ffmpeg_gpu(video_path, frame_idx)
        except Exception as e:
            cprint(f"GPU加速失败，回退到CPU: {str(e)}", 'yellow')
            return extract_frame_opencv(video_path, frame_idx)
    else:
        return extract_frame_opencv(video_path, frame_idx)

def extract_frame_ffmpeg_gpu(video_path, frame_idx):
    """
    使用FFmpeg GPU加速提取帧
    """
    # 创建临时文件
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
        temp_path = tmp_file.name
    
    try:
        # 使用FFmpeg GPU加速提取帧
        # -hwaccel cuda: 启用CUDA硬件加速
        # -ss: 设置开始时间
        # -vframes 1: 只提取1帧
        # -vf: 视频滤镜
        # -q:v 2: 高质量输出
        
        cmd = [
            'ffmpeg',
            '-hwaccel', 'cuda',  # 使用CUDA加速
            '-hwaccel_output_format', 'cuda',  # 输出格式为CUDA
            '-ss', str(frame_idx),  # 跳转到指定帧
            '-i', video_path,  # 输入文件
            '-vframes', '1',  # 只提取1帧
            '-vf', 'scale_cuda=1920:1080:format=yuv420p',  # GPU缩放
            '-q:v', '2',  # 高质量
            '-y',  # 覆盖输出文件
            temp_path
        ]
        
        # 执行命令
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg执行失败: {result.stderr}")
        
        # 读取提取的帧
        frame = cv2.imread(temp_path)
        if frame is None:
            raise RuntimeError("无法读取提取的帧")
        
        return frame
        
    finally:
        # 清理临时文件
        if os.path.exists(temp_path):
            os.unlink(temp_path)

def extract_frame_opencv(video_path, frame_idx):
    """
    使用OpenCV提取帧（CPU版本，作为备选）
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    # 获取视频信息
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

def extract_frames_batch_gpu(video_path, frame_indices, use_gpu=True, batch_size=10):
    """
    批量提取多个帧，使用GPU加速
    
    Args:
        video_path: 视频文件路径
        frame_indices: 帧索引列表
        use_gpu: 是否使用GPU
        batch_size: 批处理大小
    
    Returns:
        frames: 提取的帧列表
    """
    if use_gpu:
        try:
            return extract_frames_ffmpeg_batch(video_path, frame_indices, batch_size)
        except Exception as e:
            cprint(f"GPU批量提取失败，回退到CPU: {str(e)}", 'yellow')
            return extract_frames_opencv_batch(video_path, frame_indices)
    else:
        return extract_frames_opencv_batch(video_path, frame_indices)

def extract_frames_ffmpeg_batch(video_path, frame_indices, batch_size=10):
    """
    使用FFmpeg批量提取帧（GPU加速）
    """
    frames = []
    
    # 分批处理
    for i in range(0, len(frame_indices), batch_size):
        batch_indices = frame_indices[i:i+batch_size]
        
        # 创建临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            # 构建FFmpeg命令，提取多个帧
            filter_complex = []
            output_files = []
            
            for j, frame_idx in enumerate(batch_indices):
                output_file = os.path.join(temp_dir, f'frame_{j:04d}.jpg')
                output_files.append(output_file)
                filter_complex.append(f'[0:v]trim=start_frame={frame_idx}:end_frame={frame_idx+1},fps=1[v{j}]')
            
            # 合并filter_complex
            filter_str = ';'.join(filter_complex)
            
            # 构建输出映射
            output_mapping = []
            for j, output_file in enumerate(output_files):
                output_mapping.extend(['-map', f'[v{j}]', output_file])
            
            cmd = [
                'ffmpeg',
                '-hwaccel', 'cuda',
                '-hwaccel_output_format', 'cuda',
                '-i', video_path,
                '-filter_complex', filter_str,
                '-y'
            ] + output_mapping
            
            # 执行命令
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg批量提取失败: {result.stderr}")
            
            # 读取提取的帧
            for output_file in output_files:
                frame = cv2.imread(output_file)
                if frame is not None:
                    frames.append(frame)
                else:
                    raise RuntimeError(f"无法读取帧文件: {output_file}")
    
    return frames

def extract_frames_opencv_batch(video_path, frame_indices):
    """
    使用OpenCV批量提取帧（CPU版本）
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    try:
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                raise RuntimeError(f"无法读取第 {frame_idx} 帧")
    finally:
        cap.release()
    
    return frames

def validate_video_frames(h5_path, traj_key):
    """
    验证视频总帧数是否等于trajectory的长度
    """
    # 获取trajectory长度
    with h5py.File(h5_path, 'r') as f:
        if traj_key not in f:
            raise ValueError(f"在.h5文件中找不到 {traj_key}")
        
        traj_data = f[traj_key]
        if 'action' not in traj_data:
            raise ValueError(f"在 {traj_key} 中找不到 'action' 数据")
        
        traj_length = traj_data['action'].shape[0]
    
    # 检查两个视角的视频文件
    h5_dir = os.path.dirname(h5_path)
    view0_video = os.path.join(h5_dir, '0', f'{int(traj_key.split("_")[1])}.mp4')
    view1_video = os.path.join(h5_dir, '1', f'{int(traj_key.split("_")[1])}.mp4')
    
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

def extract_trajectory_frames(h5_path, traj_key, use_gpu=True):
    """
    提取trajectory对应的所有帧图片（GPU加速版本）
    """
    h5_dir = os.path.dirname(h5_path)
    view0_video = os.path.join(h5_dir, '0', f'{int(traj_key.split("_")[1])}.mp4')
    view1_video = os.path.join(h5_dir, '1', f'{int(traj_key.split("_")[1])}.mp4')
    
    # 获取trajectory长度和复制数据
    with h5py.File(h5_path, 'r') as f:
        traj_data = f[traj_key]
        traj_length = traj_data['action'].shape[0]
        
        # 复制所有需要的数据，避免文件句柄关闭后引用失效
        traj_data_copy = {}
        for key in traj_data.keys():
            traj_data_copy[key] = traj_data[key][()].copy()
    
    # 使用批量提取提高效率
    frame_indices = list(range(traj_length))
    
    try:
        # 批量提取两个视角的所有帧
        view0_frames = extract_frames_batch_gpu(view0_video, frame_indices, use_gpu, batch_size=20)
        view1_frames = extract_frames_batch_gpu(view1_video, frame_indices, use_gpu, batch_size=20)
        
        if len(view0_frames) != traj_length or len(view1_frames) != traj_length:
            raise RuntimeError(f"提取的帧数不匹配: view0={len(view0_frames)}, view1={len(view1_frames)}, 期望={traj_length}")
        
        return view0_frames, view1_frames, traj_data_copy
        
    except Exception as e:
        cprint(f"批量提取失败，回退到逐帧提取: {str(e)}", 'yellow')
        # 回退到逐帧提取
        view0_frames = []
        view1_frames = []
        
        for frame_idx in range(traj_length):
            try:
                frame0 = extract_frame_from_video_gpu(view0_video, frame_idx, use_gpu)
                frame1 = extract_frame_from_video_gpu(view1_video, frame_idx, use_gpu)
                
                view0_frames.append(frame0)
                view1_frames.append(frame1)
                
            except Exception as e:
                raise RuntimeError(f"提取第 {frame_idx} 帧时出错: {str(e)}")
        
        return view0_frames, view1_frames, traj_data_copy

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
    cprint(f"GPU加速: {'启用' if args.use_gpu else '禁用'}", 'yellow')
    
    # 存储所有数据 - 按episode组织
    all_episode_view0_frames = []  # 每个episode的第0帧
    all_episode_view1_frames = []  # 每个episode的第0帧
    all_episode_actions = []       # 每个episode的所有帧action数据
    all_episode_states = []        # 每个episode的第0帧action数据
    episode_ends = []              # 每个episode的有效长度
    
    try:
        with h5py.File(input_h5, 'r') as f:
            trajectory_keys = list(f.keys())
            trajectory_keys = [ traj_name for traj_name in trajectory_keys if "traj_" in traj_name]
            cprint(f"找到 {len(trajectory_keys)} 个trajectory", 'yellow')
            
            for traj_key in trajectory_keys:  # 只处理前10个用于测试
                cprint(f"处理 {traj_key}...", 'cyan')
                
                try:
                    # 验证视频帧数
                    validate_video_frames(input_h5, traj_key)
                    
                    # 提取帧和trajectory数据（使用GPU加速）
                    view0_frames, view1_frames, traj_data = extract_trajectory_frames(
                        input_h5, traj_key, use_gpu=args.use_gpu
                    )
                    
                    # 获取trajectory数据
                    actions = traj_data['action']
                    
                    # 添加到总数据中 - 按episode组织
                    all_episode_view0_frames.append(view0_frames[0])  # 只取第0帧
                    all_episode_view1_frames.append(view1_frames[0])  # 只取第0帧
                    all_episode_actions.append(actions)              # 所有帧的action
                    all_episode_states.append(actions[0, :7])       # 第0帧的action（前7列）
                    
                    # 记录每个episode的有效长度（没有padding部分的长度）
                    episode_ends.append(len(view0_frames))
                    
                    cprint(f"✓ {traj_key} 处理完成，帧数: {len(view0_frames)}", 'green')
                    
                except Exception as e:
                    cprint(f"✗ 处理 {traj_key} 时出错: {str(e)}", 'red')
                    import traceback
                    cprint(f"详细错误信息: {traceback.format_exc()}", 'red')
                    continue
        
        if not all_episode_view0_frames:
            raise RuntimeError("没有成功处理任何trajectory")
        
        cprint(f"开始读取condition数据: {args.input_condition}", 'yellow')
        
        # 读取condition数据
        condition_data = {}
        max_length = None
        
        try:
            with h5py.File(args.input_condition, 'r') as f_cond:
                # 获取max_length
                if 'meta' not in f_cond or 'max_length' not in f_cond['meta']:
                    raise ValueError("在condition文件中找不到 'meta/max_length'")
                
                max_length = int(f_cond['meta']['max_length'][()])
                cprint(f"Condition文件max_length: {max_length}", 'yellow')
                
                # 读取所有trajectory的condition数据
                for traj_key in f_cond.keys():
                    if traj_key.startswith('traj_'):
                        traj_data = f_cond[traj_key]
                        if 'action' in traj_data:
                            action_data = traj_data['action'][()]
                            # 验证维度
                            if action_data.shape != (max_length, 8):
                                raise ValueError(f"Trajectory {traj_key} 的action维度不正确: {action_data.shape}, 期望: ({max_length}, 8)")
                            condition_data[traj_key] = action_data
                
                cprint(f"成功读取 {len(condition_data)} 个trajectory的condition数据", 'green')
                
        except Exception as e:
            cprint(f"读取condition文件时出错: {str(e)}", 'red')
            raise
        
        cprint(f"开始保存到zarr文件: {output_zarr}", 'yellow')
        
        # 创建zarr文件
        zarr_root = zarr.group(output_zarr)
        zarr_data = zarr_root.create_group('data')
        zarr_meta = zarr_root.create_group('meta')
        
        # 计算episode数量
        num_episodes = len(all_episode_view0_frames)
        cprint(f"Episode数量: {num_episodes}", 'yellow')
        
        # 获取图像尺寸
        img_height, img_width, img_channels = all_episode_view0_frames[0].shape
        
        # 1. 处理图像数据 - 每个episode的第0帧，维度为 (num_episodes, channel, width, height)
        img1_arrays = np.zeros((num_episodes, img_channels, img_width, img_height), dtype=np.uint8)
        img2_arrays = np.zeros((num_episodes, img_channels, img_width, img_height), dtype=np.uint8)
        
        # 填充图像数据
        for ep_idx in range(num_episodes):
            img1_arrays[ep_idx] = cv2.cvtColor(all_episode_view0_frames[ep_idx], cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
            img2_arrays[ep_idx] = cv2.cvtColor(all_episode_view1_frames[ep_idx], cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        
        # 2. 处理state数据 - 每个episode第0帧的action，shape [num_episodes, 7]
        state_arrays = np.array(all_episode_states, dtype=np.float32)
        
        # 3. 处理action数据 - 每个episode的所有帧的action，需要padding到max_length
        # action_arrays shape: [num_episodes, max_length, 8]
        action_arrays = np.full((num_episodes, max_length, 8), -1, dtype=np.float32)
        
        for ep_idx, actions in enumerate(all_episode_actions):
            T = actions.shape[0]  # 当前episode的帧数
            
            if actions.shape[1] >= 8:
                # 填充action数据，不足的部分用-1填充
                action_arrays[ep_idx, :T, :8] = actions[:, :8]
            else:
                raise RuntimeError(f"Episode {ep_idx} 的action列数不足8. 请检查.h5原始数据文件")
        
        # 4. 处理MAS数据 - 每个episode第0帧对应的condition
        # MAS_arrays shape: [num_episodes, max_length, 8]
        MAS_arrays = np.zeros((num_episodes, max_length, 8), dtype=np.float32)
        
        for ep_idx, traj_key in enumerate(trajectory_keys[:num_episodes]):
            if traj_key in condition_data:
                MAS_arrays[ep_idx] = condition_data[traj_key]
            else:
                raise ValueError(f"找不到episode {ep_idx} 对应的trajectory {traj_key} 的condition数据")
        
        # 设置压缩器
        compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
        
        print(f"img1 shape: {img1_arrays.shape}")
        print(f"img2 shape: {img2_arrays.shape}")
        print(f"state shape: {state_arrays.shape}")
        print(f"action shape: {action_arrays.shape}")
        print(f"MAS shape: {MAS_arrays.shape}")
        
        # 设置chunk大小
        img_chunk_size = (100, img_channels, img_width, img_height)
        state_chunk_size = (100, 7)
        action_chunk_size = (100, max_length, 8)
        MAS_chunk_size = (100, max_length, 8)
        
        # 保存数据到data组
        zarr_data.create_dataset('img1', data=img1_arrays, chunks=img_chunk_size, 
                                dtype='uint8', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('img2', data=img2_arrays, chunks=img_chunk_size, 
                                dtype='uint8', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('state', data=state_arrays, chunks=state_chunk_size, 
                                dtype='float32', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('action', data=action_arrays, chunks=action_chunk_size, 
                                dtype='float32', overwrite=True, compressor=compressor)
        zarr_data.create_dataset('MAS', data=MAS_arrays, chunks=MAS_chunk_size, 
                                dtype='float32', overwrite=True, compressor=compressor)
        
        # 保存episode信息到meta组
        # episode_ends: 每个episode的有效长度（没有padding部分的长度）
        episode_ends_arrays = np.array(episode_ends)
        zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, 
                                dtype='int64', overwrite=True, compressor=compressor)
        
        cprint(f'-'*50, 'cyan')
        cprint(f'img1 shape: {img1_arrays.shape}, range: [{np.min(img1_arrays)}, {np.max(img1_arrays)}]', 'green')
        cprint(f'img2 shape: {img2_arrays.shape}, range: [{np.min(img2_arrays)}, {np.max(img2_arrays)}]', 'green')
        cprint(f'state shape: {state_arrays.shape}, range: [{np.min(state_arrays)}, {np.max(state_arrays)}]', 'green')
        cprint(f'action shape: {action_arrays.shape}, range: [{np.min(action_arrays)}, {np.max(action_arrays)}]', 'green')
        cprint(f'MAS shape: {MAS_arrays.shape}, range: [{np.min(MAS_arrays)}, {np.max(MAS_arrays)}]', 'green')
        cprint(f'episode_ends (有效长度): {episode_ends_arrays}', 'green')
        cprint(f'成功保存zarr文件到 {output_zarr}', 'green')
        
    except Exception as e:
        cprint(f"处理过程中出错: {str(e)}", 'red')
        raise
    
    finally:
        # 清理内存
        if 'all_episode_view0_frames' in locals():
            del all_episode_view0_frames, all_episode_view1_frames, all_episode_actions, all_episode_states
        if 'zarr_root' in locals():
            del zarr_root, zarr_data, zarr_meta

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default="../demo_0828/PlugCharger-v1/motionplanning/action_normed.h5",  help='输入归一化后的原始.h5文件路径')
    parser.add_argument('--output', type=str, default="./output_zarr/test_mam_gpu.zarr", help='输出.zarr文件路径')
    parser.add_argument('--input_condition', type=str, default="./output/0830_test_PlugCharger_local_planner_padding_1.h5",  help='输入mask action condition .h5文件')
    parser.add_argument('--use_gpu', type=bool, default=True, help='是否使用GPU加速视频帧提取')
    
    args = parser.parse_args()
    main(args) 