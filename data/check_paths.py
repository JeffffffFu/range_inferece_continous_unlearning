#!/usr/bin/env python3
"""
检查数据集和tokenizer的下载路径
"""

import os
import sys
from datasets import load_dataset
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

def check_download_paths():
    """检查各种下载路径"""
    print("=" * 60)
    print("数据集和模型下载路径检查")
    print("=" * 60)
    
    # 1. 检查当前工作目录
    print(f"1. 当前工作目录: {os.getcwd()}")
    
    # 2. 检查图像数据集路径
    image_data_path = os.getcwd() + '/data'
    print(f"2. 图像数据集路径: {image_data_path}")
    print(f"   路径是否存在: {os.path.exists(image_data_path)}")
    
    # 3. 检查Hugging Face缓存路径
    from datasets import config
    hf_cache_dir = config.HF_DATASETS_CACHE
    print(f"3. Hugging Face数据集缓存路径: {hf_cache_dir}")
    print(f"   路径是否存在: {os.path.exists(hf_cache_dir)}")
    
    # 4. 检查Hugging Face Hub缓存路径
    try:
        from huggingface_hub import HF_HUB_CACHE
        hub_cache_dir = HF_HUB_CACHE
        print(f"4. Hugging Face Hub缓存路径: {hub_cache_dir}")
        print(f"   路径是否存在: {os.path.exists(hub_cache_dir)}")
    except ImportError:
        print("4. 无法获取Hugging Face Hub缓存路径")
    
    # 5. 检查环境变量
    print(f"5. 环境变量:")
    print(f"   HF_HOME: {os.environ.get('HF_HOME', '未设置')}")
    print(f"   HF_DATASETS_CACHE: {os.environ.get('HF_DATASETS_CACHE', '未设置')}")
    print(f"   TRANSFORMERS_CACHE: {os.environ.get('TRANSFORMERS_CACHE', '未设置')}")
    
    # 6. 尝试下载SST5数据集并查看实际路径
    print(f"\n6. 尝试下载SST5数据集...")
    try:
        dataset = load_dataset("SetFit/sst5", download_mode="force_redownload")
        print(f"   SST5数据集下载成功!")
        
        # 获取数据集的实际路径
        dataset_path = dataset['train']._info.dataset_size
        print(f"   数据集信息: {dataset['train']._info}")
        
        # 查找SST5数据集文件
        sst5_cache_dir = os.path.join(hf_cache_dir, "setfit___sst5")
        if os.path.exists(sst5_cache_dir):
            print(f"   SST5数据集缓存目录: {sst5_cache_dir}")
            print(f"   目录内容:")
            for item in os.listdir(sst5_cache_dir):
                item_path = os.path.join(sst5_cache_dir, item)
                if os.path.isdir(item_path):
                    print(f"     📁 {item}/")
                    for subitem in os.listdir(item_path):
                        print(f"       📄 {subitem}")
                else:
                    print(f"     📄 {item}")
        
    except Exception as e:
        print(f"   下载失败: {e}")
    
    # 7. 尝试下载tokenizer并查看实际路径
    print(f"\n7. 尝试下载GPT-2 tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        print(f"   GPT-2 tokenizer下载成功!")
        
        # 获取tokenizer的实际路径
        tokenizer_path = tokenizer.name_or_path
        print(f"   Tokenizer路径: {tokenizer_path}")
        
        # 查找tokenizer缓存文件
        gpt2_cache_dir = os.path.join(hub_cache_dir, "models--gpt2--main")
        if os.path.exists(gpt2_cache_dir):
            print(f"   GPT-2模型缓存目录: {gpt2_cache_dir}")
            print(f"   目录内容:")
            for item in os.listdir(gpt2_cache_dir):
                item_path = os.path.join(gpt2_cache_dir, item)
                if os.path.isdir(item_path):
                    print(f"     📁 {item}/")
                    for subitem in os.listdir(item_path):
                        print(f"       📄 {subitem}")
                else:
                    print(f"     📄 {item}")
        
    except Exception as e:
        print(f"   下载失败: {e}")

def show_directory_structure(path, max_depth=2, current_depth=0):
    """显示目录结构"""
    if current_depth >= max_depth:
        return
    
    if not os.path.exists(path):
        return
    
    indent = "  " * current_depth
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            print(f"{indent}📁 {item}/")
            show_directory_structure(item_path, max_depth, current_depth + 1)
        else:
            print(f"{indent}📄 {item}")

if __name__ == "__main__":
    check_download_paths()
    
    print(f"\n" + "=" * 60)
    print("缓存目录结构概览")
    print("=" * 60)
    
    # 显示Hugging Face缓存目录结构
    from datasets import config
    hf_cache_dir = config.HF_DATASETS_CACHE
    if os.path.exists(hf_cache_dir):
        print(f"\nHugging Face数据集缓存目录结构:")
        show_directory_structure(hf_cache_dir, max_depth=1)
    
    try:
        from huggingface_hub import HF_HUB_CACHE
        hub_cache_dir = HF_HUB_CACHE
        if os.path.exists(hub_cache_dir):
            print(f"\nHugging Face Hub缓存目录结构:")
            show_directory_structure(hub_cache_dir, max_depth=1)
    except ImportError:
        pass 