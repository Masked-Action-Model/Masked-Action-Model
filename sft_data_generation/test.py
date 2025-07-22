import h5py
file_path = '../se3_raw/DrawSVG-v1/motionplanning/20250716_200411_mask.h5'
with h5py.File(file_path, 'r') as h5file:
    # 获取根目录下的所有键
    keys = list(h5file.keys())
    print("文件中的键有:", keys)
    
    # 遍历每个键
    for key in keys:
        item = h5file[key]
        
        # 判断是组还是数据集
        if isinstance(item, h5py.Group):
            print(f"\n'{key}' 是一个组，包含以下内容:")
            group_keys = list(item.keys())
            print(group_keys)
            
            # 递归查看子组或数据集
            for sub_key in group_keys:
                sub_item = item[sub_key]
                if isinstance(sub_item, h5py.Group):
                    print(f"  - '{sub_key}' 是子组")
                else:  # 是数据集
                    print(f"  - '{sub_key}' 是数据集，形状: {sub_item.shape}, 数据类型: {sub_item.dtype}")
                    # 读取数据集内容
                    data = sub_item[:]
                    print(f"  - 数据内容示例: {data[:10]}")  # 显示前10个元素
                    
        else:  # 是数据集
            print(f"\n'{key}' 是数据集，形状: {item.shape}, 数据类型: {item.dtype}")
            data = item[:]
            print(f"  - 数据内容示例: {data[:10]}")