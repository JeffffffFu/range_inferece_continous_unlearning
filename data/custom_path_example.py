#!/usr/bin/env python3
"""
自定义缓存路径使用示例
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.load_data import get_data
from data.sst5_data import load_sst5_data, set_cache_paths, get_cache_info

def example_custom_paths():
    """自定义路径使用示例"""
    print("=" * 60)
    print("自定义缓存路径使用示例")
    print("=" * 60)
    
    # 方法1: 使用环境变量设置
    print("\n1. 使用环境变量设置缓存路径")
    custom_cache = "D:/my_cache/huggingface"
    set_cache_paths(cache_dir=custom_cache)
    
    # 使用get_data函数
    try:
        train_set, test_set = get_data("sst5")
        print(f"✓ 使用get_data成功加载，训练集大小: {len(train_set)}")
    except Exception as e:
        print(f"✗ 使用get_data失败: {e}")
    
    # 方法2: 直接在函数中指定路径
    print("\n2. 直接在函数中指定路径")
    try:
        train_set, test_set, tokenizer = load_sst5_data(
            tokenizer_name="gpt2",
            max_length=128,
            merge_val_to_train=True,
            cache_dir="E:/my_other_cache/huggingface"  # 指定不同的缓存路径
        )
        print(f"✓ 使用load_sst5_data成功加载，训练集大小: {len(train_set)}")
    except Exception as e:
        print(f"✗ 使用load_sst5_data失败: {e}")
    
    # 方法3: 分别设置数据集和模型缓存路径
    print("\n3. 分别设置数据集和模型缓存路径")
    try:
        train_set, test_set, tokenizer = load_sst5_data(
            tokenizer_name="gpt2",
            max_length=128,
            merge_val_to_train=True,
            data_cache_dir="F:/datasets_cache",
            model_cache_dir="F:/models_cache"
        )
        print(f"✓ 分别设置路径成功加载，训练集大小: {len(train_set)}")
    except Exception as e:
        print(f"✗ 分别设置路径失败: {e}")

def test_different_cache_locations():
    """测试不同的缓存位置"""
    print("\n" + "=" * 60)
    print("测试不同的缓存位置")
    print("=" * 60)
    
    # 测试不同的缓存目录
    cache_locations = [
        "D:/huggingface_cache",
        "E:/my_data/hf_cache", 
        "./local_cache",  # 相对路径
        os.path.expanduser("~/my_hf_cache")  # 用户目录
    ]
    
    for i, cache_dir in enumerate(cache_locations, 1):
        print(f"\n测试位置 {i}: {cache_dir}")
        
        # 确保目录存在
        os.makedirs(cache_dir, exist_ok=True)
        
        try:
            # 设置缓存路径
            set_cache_paths(cache_dir=cache_dir)
            
            # 加载数据集
            train_set, test_set, tokenizer = load_sst5_data(
                tokenizer_name="gpt2",
                max_length=128,
                merge_val_to_train=True
            )
            
            print(f"✓ 成功加载，训练集大小: {len(train_set)}")
            
            # 检查文件是否真的下载到了指定位置
            expected_data_dir = os.path.join(cache_dir, "datasets", "setfit___sst5")
            expected_model_dir = os.path.join(cache_dir, "models", "models--gpt2--main")
            
            print(f"  数据集目录存在: {os.path.exists(expected_data_dir)}")
            print(f"  模型目录存在: {os.path.exists(expected_model_dir)}")
            
        except Exception as e:
            print(f"✗ 失败: {e}")

def show_cache_structure(cache_dir):
    """显示缓存目录结构"""
    print(f"\n缓存目录结构: {cache_dir}")
    if not os.path.exists(cache_dir):
        print("  目录不存在")
        return
    
    for root, dirs, files in os.walk(cache_dir):
        level = root.replace(cache_dir, '').count(os.sep)
        indent = '  ' * level
        print(f"{indent}📁 {os.path.basename(root)}/")
        subindent = '  ' * (level + 1)
        for file in files[:5]:  # 只显示前5个文件
            print(f"{subindent}📄 {file}")
        if len(files) > 5:
            print(f"{subindent}... 还有 {len(files) - 5} 个文件")

if __name__ == "__main__":
    # 显示当前缓存信息
    print("当前缓存信息:")
    get_cache_info()
    
    # 运行示例
    example_custom_paths()
    
    # 测试不同缓存位置
    test_different_cache_locations()
    
    # 显示最终缓存结构
    print("\n" + "=" * 60)
    print("最终缓存结构")
    print("=" * 60)
    
    # 检查默认缓存目录
    from datasets import config
    default_cache = config.HF_DATASETS_CACHE
    show_cache_structure(default_cache)
    
    # 检查自定义缓存目录
    custom_cache = "D:/my_cache/huggingface"
    if os.path.exists(custom_cache):
        show_cache_structure(custom_cache) 