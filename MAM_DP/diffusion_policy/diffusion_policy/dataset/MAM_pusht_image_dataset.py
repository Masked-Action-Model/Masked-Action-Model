from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer

class MAMDataset(BaseImageDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None
            ):
        
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['img1', 'img2', 'state', 'action', 'condition'])
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)
          
        self.sampler = SequenceSampler(         #形状变化
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask) 
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs): #condition已经归一化过了
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state']    
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        #normalizer['image'] = get_image_range_normalizer()
        normalizer['img1'] = get_image_range_normalizer()
        normalizer['img2'] = get_image_range_normalizer()
        
        # Add normalizer for condition - identity normalizer since it's generated randomly
        from diffusion_policy.model.common.normalizer import SingleFieldLinearNormalizer
        normalizer['condition'] = SingleFieldLinearNormalizer.create_identity()
        
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['state'].astype(np.float32)
        
        # 处理两张图片
        img1 = sample['img1'].astype(np.float32) / 255.0  # T,3,512,512
        img2 = sample['img2'].astype(np.float32) / 255.0  # T,3,512,512

        T = len(agent_pos)
        condition = sample['condition']
        condition = condition.reshape(T, -1)  # (T, 800)
        
        data = {
            'obs': {
                'img1': img1,  # T,3,512,512
                'img2': img2,  # T,3,512,512
                'agent_pos': agent_pos,
                'condition': condition,
            },
            'action': sample['action'].astype(np.float32)
        }
        
        return data
        #print(f'data_agent_pos.shape={agent_pos.shape}')
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx) #根据给定的 idx（样本编号），从 SequenceSampler 采样器中获取一个长度为 horizon 的时序片段（已补齐），返回一个 dict，包含如 'image'、'state'、'action'、'condition' 等键，每个键的值都是 numpy 数组，时间维度为
        data = self._sample_to_data(sample) #对采样得到的原始 dict 做进一步处理，比如图像通道换位、归一化、condition 展平等，整理成模型需要的格式，依然是 dict，值为 numpy 数组。
        torch_data = dict_apply(data, torch.from_numpy) #返回最终的单条样本（无 batch 维），每个 key 都是 torch.Tensor，时间维度为 T=horizon。DataLoader 会自动堆叠多个样本形成 batch。
        return torch_data


def test():
    import os
    zarr_path = os.path.expanduser('~/dev/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr')
    dataset = PushTImageDataset(zarr_path, horizon=16)

    # from matplotlib import pyplot as plt
    # normalizer = dataset.get_normalizer()
    # nactions = normalizer['action'].normalize(dataset.replay_buffer['action'])
    # diff = np.diff(nactions, axis=0)
    # dists = np.linalg.norm(np.diff(nactions, axis=0), axis=-1)
