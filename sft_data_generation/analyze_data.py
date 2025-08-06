import h5py
import numpy as np
from pathlib import Path

def analyze_h5_file(h5_file_path):
    """分析h5文件中的action数据"""
    results = {
        'max_action_shape': None,
        'trajectories': {},
        'total_trajectories': 0,
        'max_trajectory_length': 0
    }
    
    try:
        with h5py.File(h5_file_path, 'r') as f:
            if 'trajectories' not in f:
                print(f"文件中未找到trajectories组")
                return results
            
            trajectories_group = f['trajectories']
            
            for traj_key in trajectories_group.keys():
                traj_group = trajectories_group[traj_key]
                
                if 'action' in traj_group:
                    action_data = traj_group['action']
                    action_shape = action_data.shape
                    
                    # 更新最大action形状
                    if results['max_action_shape'] is None:
                        results['max_action_shape'] = action_shape
                    else:
                        # 比较每个维度，取最大值
                        max_shape = tuple(max(a, b) for a, b in zip(results['max_action_shape'], action_shape))
                        results['max_action_shape'] = max_shape
                    
                    # 更新最大轨迹长度
                    if action_shape[0] > results['max_trajectory_length']:
                        results['max_trajectory_length'] = action_shape[0]
                    
                    results['trajectories'][traj_key] = {
                        'action_shape': action_shape,
                        'action_dtype': str(action_data.dtype)
                    }
                    
                    print(f"轨迹 {traj_key}: action shape = {action_shape}, dtype = {action_data.dtype}")
                
                results['total_trajectories'] += 1
    
    except Exception as e:
        print(f"读取h5文件失败: {e}")
    
    return results

def main():
    # 指定h5文件路径
    #h5_file_path = "../Data/arlen_data/demos/PickCube-v1/motionplanning/20250716_184350_mask.h5"
    h5_file_path = "../Data/arlen_data/demos/DrawTriangle-v1/motionplanning/20250716_184925_mask.h5"
    print(f"开始分析h5文件: {h5_file_path}")
    
    if not Path(h5_file_path).exists():
        print(f"文件不存在: {h5_file_path}")
        return
    
    results = analyze_h5_file(h5_file_path)
    
    print("\n" + "="*50)
    print("H5文件分析结果:")
    print("="*50)
    print(f"最大action形状: {results['max_action_shape']}")
    print(f"最大轨迹长度: {results['max_trajectory_length']}")
    print(f"总轨迹数量: {results['total_trajectories']}")
    
    
if __name__ == '__main__':
    main()