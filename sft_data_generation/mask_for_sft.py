import h5py
import numpy as np
import argparse
import os
import json

class MaskSFT:
    def __init__(self, input_path, output_path, mask_type, retain_ratio=None, size=1, normalize=False, mask_seq_len=20):
        self.input_path = input_path
        self.output_path = output_path
        self.mask_type = mask_type
        self.retain_ratio = retain_ratio
        self.size = size
        self.normalize = normalize
        self.mask_seq_len = mask_seq_len

    def mask_action(self, action, retain_ratio=None):
        # action: shape (N, 8)
        mask = np.ones(action.shape[1], dtype=bool)
        if self.mask_type == '2D_video_trajectory':
            # 保留x, y, time step (0,1,7)
            mask[[0,1,7]] = False
            action[:, mask] = -1
        elif self.mask_type == '2D_image_trajectory':
            # 只保留x, y (0,1)
            mask = np.ones(action.shape[1], dtype=bool)
            mask[[0,1]] = False
            action[:, mask] = -1
        elif self.mask_type == 'pose_AnyGrasp':
            # 保留任意一个的x,y,z,dx,dy,dz,gripper (0-6)
            n = action.shape[0]
            retain_num = 1
            idx = np.arange(n)
            np.random.shuffle(idx)
            retain_idx = idx[:retain_num]
            action[:, :] = -1
            action[retain_idx, :7] = self._original_action[retain_idx, :7]
        elif self.mask_type == 'pose_motion_planning':
            # 保留retain_ratio比例的x,y,z,dx,dy,dz,gripper(0-6)
            if retain_ratio is None:
                raise ValueError('pose_motion_planning类型需要retain_ratio参数')
            n = action.shape[0]
            retain_num = int(n * retain_ratio)
            idx = np.arange(n)
            np.random.shuffle(idx)
            retain_idx = idx[:retain_num]
            action[:, :] = -1
            action[retain_idx, :7] = self._original_action[retain_idx, :7]
        elif self.mask_type == 'points':
            # 保留retain_ratio比例的x, y (0,1)
            if retain_ratio is None:
                raise ValueError('points类型需要retain_ratio参数')
            n = action.shape[0]
            retain_num = int(n * retain_ratio)
            idx = np.arange(n)
            np.random.shuffle(idx)
            retain_idx = idx[:retain_num]
            action[:, :] = -1
            action[retain_idx, 0:2] = self._original_action[retain_idx, 0:2]
        elif self.mask_type == 'auto_regressive':
            # 随机选取一个点(i, j)，将第i行j及其后所有列mask，第i+1及其后所有行全部mask
            n, m = action.shape
            if n == 0 or m == 0:
                return action
            i = np.random.randint(0, n)
            j = np.random.randint(0, m)
            action[i, j:] = -1
            if i + 1 < n:
                action[i+1:, :] = -1
        elif self.mask_type == 'local_planner':
            # 保留所有time（第7列）数据，mask掉连续长度为mask_seq_len的任意一个子序列（0-6列），其余部分都保留
            n = action.shape[0]
            # 先保留所有原始数据
            action[:, :] = self._original_action[:, :]
            
            # 容错机制：确保mask_seq_len在合理范围内
            if self.mask_seq_len <= 0:
                # 如果mask_seq_len <= 0，则不mask任何数据
                print("mask_seq_len设置为负数当前mask操作无效, 保留原始数据")
                pass
            elif self.mask_seq_len >= n:
                # 如果mask_seq_len >= 序列长度，则mask所有0-6列数据
                print("mask_seq_len设置为大于序列长度, 请设置小于序列长度的mask序列长度")
                pass
            else:
                # 随机选择一个起始位置，mask掉连续长度为mask_seq_len的子序列
                max_start = n - self.mask_seq_len
                start_idx = np.random.randint(0, max_start + 1)
                end_idx = start_idx + self.mask_seq_len
                # 确保范围在原始序列之内
                start_idx = max(0, min(start_idx, n - self.mask_seq_len))
                end_idx = start_idx + self.mask_seq_len
                action[start_idx:end_idx, 0:7] = -1
        elif self.mask_type == 'random_mask':
            # 保留retain_ratio比例的随机mask
            if retain_ratio is None:
                raise ValueError('random_mask类型需要retain_ratio参数')
            n,m = action.shape
            idx = []
            retain_num = int(n * m * retain_ratio)
            for x in range(n):
                for y in range(m):
                    idx.append((x,y))
            np.random.shuffle(idx)
            retain_idx = idx[:retain_num]
            action[:, :] = -1
            for (x,y) in retain_idx:
                action[x,y] = self._original_action[x,y]
        return action

    def run(self):
        # 如果需要归一化，先归一化并保存
        norm_json_path = os.path.splitext(self.input_path)[0] + '_norm.json'
        normed_h5_path = self.input_path
        if self.normalize:
            normed_h5_path = os.path.splitext(self.input_path)[0] + '_normed.h5'
            mins, maxs = normalize_actions_to_new_h5(self.input_path, normed_h5_path, norm_json_path)
        self.process(normed_h5_path)

    def process(self, input_path=None):
        retain_ratio = None
        if self.mask_type in ['pose_AnyGrasp', 'points', 'pose_motion_planning', 'random_mask']:
            retain_ratio = self.retain_ratio
        single_file_types = ['2D_video_trajectory', '2D_image_trajectory']
        num_files = 1 if self.mask_type in single_file_types else self.size
        unique_masks = set()
        if input_path is None:
            input_path = self.input_path
        for idx in range(num_files):
            if num_files == 1:
                output_path = self.output_path
            else:
                base, ext = os.path.splitext(self.output_path)
                output_path = f"{base}_{idx}{ext}"
            with h5py.File(input_path, 'r') as f_in, h5py.File(output_path, 'w') as f_out:
                for traj_key in f_in['trajectories']:
                    grp_in = f_in['trajectories'][traj_key]
                    grp_out = f_out.require_group(f'trajectories/{traj_key}')
                    for dset_key in grp_in:
                        data = grp_in[dset_key][()]
                        if dset_key == 'action':
                            if self.mask_type in ['pose_AnyGrasp', 'points', 'pose_motion_planning', 'auto_regressive', 'random_mask', 'local_planner']:
                                self._original_action = data.copy()
                                max_retry = 1000
                                for _ in range(max_retry):
                                    masked = self.mask_action(data.copy(), retain_ratio=retain_ratio)
                                    mask_flag = (masked == -1)
                                    mask_hash = mask_flag.tobytes()
                                    if mask_hash not in unique_masks:
                                        unique_masks.add(mask_hash)
                                        data = masked
                                        break
                                else:
                                    raise RuntimeError('无法生成足够多唯一的mask样本, 请减少size或调整retain_ratio')
                            else:
                                data = self.mask_action(data)
                            if data.shape[1] > 7:
                                # 将mask(-1)值放在最后面，其余按第七列（时间序列）从小到大排序
                                mask_neg_one = data[:, 7] == -1
                                non_neg_one_idx = np.where(~mask_neg_one)[0]
                                neg_one_idx = np.where(mask_neg_one)[0]
                                
                                if len(non_neg_one_idx) > 0:
                                    sort_idx = np.argsort(data[non_neg_one_idx, 7])
                                    non_neg_one_sorted = non_neg_one_idx[sort_idx]
                                    final_idx = np.concatenate([non_neg_one_sorted, neg_one_idx])
                                else:
                                    final_idx = np.arange(len(data))
                                data = data[final_idx]
                        grp_out.create_dataset(dset_key, data=data, compression="gzip")


def normalize_actions(input_path, output_path):
    """
    归一化所有trajectory下group的action的0-6列
    """
    mins = np.full(7, np.inf)
    maxs = np.full(7, -np.inf)
    # 先统计全局min/max
    with h5py.File(input_path, 'r') as f:
        for traj_key in f['trajectories']:
            grp = f['trajectories'][traj_key]
            if 'action' in grp:
                data = grp['action'][()]
                mins = np.minimum(mins, data[:, :7].min(axis=0))
                maxs = np.maximum(maxs, data[:, :7].max(axis=0))
    # 保存max/min到txt
    norm_info = {'min': mins.tolist(), 'max': maxs.tolist()}
    txt_path = os.path.splitext(output_path)[0] + '_norm.txt'
    with open(txt_path, 'w') as f:
        json.dump(norm_info, f, indent=2)
    return mins, maxs

def apply_normalize(data, mins, maxs):
    # 归一化到(-1,1]
    # x_norm = 2 * (x - min) / (max - min + 1e-8) - 1 + 1e-8
    normed = data.copy()
    for i in range(6):
        normed[:, i] = 2 * (data[:, i] - mins[i]) / (maxs[i] - mins[i] + 1e-8) - 1 + 1e-8
    return normed

# 归一化到新h5并保存max/min到json

def normalize_actions_to_new_h5(input_path, output_path, json_path):
    """
    归一化所有trajectory下group的action的0-5列，区间(-1,1]，写入新h5，max/min写入json。
    """
    mins = np.full(6, np.inf)
    maxs = np.full(6, -np.inf)
    with h5py.File(input_path, 'r') as f:
        for traj_key in f['trajectories']:
            grp = f['trajectories'][traj_key]
            if 'action' in grp:
                data = grp['action'][()]
                mins = np.minimum(mins, data[:, :6].min(axis=0))
                maxs = np.maximum(maxs, data[:, :6].max(axis=0))
    norm_info = {'min': mins.tolist(), 'max': maxs.tolist()}
    with open(json_path, 'w') as f:
        json.dump(norm_info, f, indent=2)
    # 写入归一化h5
    with h5py.File(input_path, 'r') as f_in, h5py.File(output_path, 'w') as f_out:
        for traj_key in f_in['trajectories']:
            grp_in = f_in['trajectories'][traj_key]
            grp_out = f_out.require_group(f'trajectories/{traj_key}')
            for dset_key in grp_in:
                data = grp_in[dset_key][()]
                if dset_key == 'action':
                    data = apply_normalize(data, mins, maxs)
                grp_out.create_dataset(dset_key, data=data, compression="gzip")
    return mins, maxs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default="/data1/tangjielong/VLA_project/se3_raw/StackPyramid-v1/motionplanning/20250716_205039_mask.h5", help='输入.h5文件路径')
    parser.add_argument('--output', type=str,  default="./test.h5", help='输出.h5文件路径')
    parser.add_argument('--mask_type', type=str, default="local_planner", help='mask类型')
    parser.add_argument('--retain_ratio', type=float, default=0.2, help='pose_AnyGrasp时保留比例')
    parser.add_argument('--size', type=int, default=5, help='生成h5文件数量')
    parser.add_argument('--normalize', action='store_true', help='是否对原始h5归一化')
    parser.add_argument('--mask_seq_len', type=int, default=20, help='local_planner类型时连续mask序列长度')
    args = parser.parse_args()

    masker = MaskSFT(args.input, args.output, args.mask_type, args.retain_ratio, args.size, args.normalize, args.mask_seq_len)
    masker.run()

if __name__ == '__main__':
    main()
