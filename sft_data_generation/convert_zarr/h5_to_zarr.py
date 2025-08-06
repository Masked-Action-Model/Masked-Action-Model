import h5py
import zarr
import numpy as np
import argparse
from pathlib import Path

def convert_h5_to_zarr(h5_file_path, zarr_file_path, compression='gzip'):
    """将h5文件转换为zarr 2.0格式"""
    
    # 创建zarr存储，指定版本为2
    store = zarr.DirectoryStore(zarr_file_path)
    root = zarr.group(store=store, overwrite=True)
    root.attrs['zarr_format'] = 2
    
    def copy_group(h5_group, zarr_group):
        """递归复制h5组到zarr组"""
        for key in h5_group.keys():
            item = h5_group[key]
            
            if isinstance(item, h5py.Group):
                # 创建子组
                sub_group = zarr_group.create_group(key)
                # 复制属性
                for attr_key, attr_value in item.attrs.items():
                    sub_group.attrs[attr_key] = attr_value
                # 递归复制子组内容
                copy_group(item, sub_group)
                
            elif isinstance(item, h5py.Dataset):
                # 复制数据集
                data = item[()]
                dataset = zarr_group.create_dataset(
                    key, 
                    data=data, 
                    compression=compression,
                    chunks=True
                )
                # 复制属性
                for attr_key, attr_value in item.attrs.items():
                    dataset.attrs[attr_key] = attr_value
                
                print(f"转换数据集: {key}, shape: {data.shape}, dtype: {data.dtype}")
    
    try:
        with h5py.File(h5_file_path, 'r') as h5_file:
            print(f"开始转换: {h5_file_path} -> {zarr_file_path}")
            
            # 复制根级别属性
            for attr_key, attr_value in h5_file.attrs.items():
                root.attrs[attr_key] = attr_value
            
            # 复制所有组和数据集
            copy_group(h5_file, root)
            
        print(f"转换完成！输出文件: {zarr_file_path}")
        
        # 验证转换结果
        verify_conversion(h5_file_path, zarr_file_path)
        
    except Exception as e:
        print(f"转换失败: {e}")
        raise

def verify_conversion(h5_file_path, zarr_file_path):
    """验证转换结果"""
    print("\n验证转换结果...")
    
    with h5py.File(h5_file_path, 'r') as h5_file:
        zarr_root = zarr.open(zarr_file_path, mode='r')
        
        def compare_structure(h5_group, zarr_group, path=""):
            """比较h5和zarr的结构"""
            for key in h5_group.keys():
                current_path = f"{path}/{key}" if path else key
                h5_item = h5_group[key]
                
                if isinstance(h5_item, h5py.Group):
                    if key in zarr_group:
                        compare_structure(h5_item, zarr_group[key], current_path)
                    else:
                        print(f"警告: zarr中缺少组 {current_path}")
                        
                elif isinstance(h5_item, h5py.Dataset):
                    if key in zarr_group:
                        h5_data = h5_item[()]
                        zarr_data = zarr_group[key][()]
                        
                        if np.array_equal(h5_data, zarr_data):
                            print(f"✓ 数据集 {current_path} 验证通过")
                        else:
                            print(f"✗ 数据集 {current_path} 数据不匹配")
                    else:
                        print(f"警告: zarr中缺少数据集 {current_path}")
        
        compare_structure(h5_file, zarr_root)

def main():
    parser = argparse.ArgumentParser(description='将h5文件转换为zarr 2.0格式')
    parser.add_argument('--input', type=str, required=True, help='输入h5文件路径')
    parser.add_argument('--output', type=str, required=True, help='输出zarr文件路径')
    parser.add_argument('--compression', type=str, default='gzip', 
                       choices=['gzip', 'lz4', 'zstd', 'blosc'], 
                       help='压缩算法')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"错误: 输入文件不存在 {input_path}")
        return
    
    if output_path.exists():
        print(f"警告: 输出路径已存在，将被覆盖 {output_path}")
    
    convert_h5_to_zarr(str(input_path), str(output_path), args.compression)

if __name__ == '__main__':
    main()