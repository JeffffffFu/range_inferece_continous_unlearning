import numpy as np
import matplotlib.pyplot as plt
import os
import random
from collections import defaultdict


def visualize_finetune_results(args, trial=0):
    """
    可视化finetune.py保存的结果
    
    Args:
        args: 参数字典，包含net_name, dataset_name, proportion_of_group_unlearn等
        trial: trial编号，默认为0
    """
    # 构建保存路径
    save_path = os.getcwd() + f"/save3/continuous_finetune/{args['net_name']}/{args['dataset_name']}/0.01/target/{trial}"
    
    print(f"Loading data from: {save_path}")
    
    # 1. 读取breakpoint索引
    breakpoint_indices_file = f"{save_path}/breakpoint_indices.npy"
    if not os.path.exists(breakpoint_indices_file):
        print(f"Error: {breakpoint_indices_file} not found!")
        return
    
    breakpoint_indices_dict = np.load(breakpoint_indices_file, allow_pickle=True).item()
    print(f"  -> Loaded breakpoint indices for {len(breakpoint_indices_dict)} samples")
    
    # 2. 找到只有一个insert和一个remove的样本
    target_samples = []
    for sample_id, bp_info in breakpoint_indices_dict.items():
        num_insert = len(bp_info['insert'])
        num_remove = len(bp_info['remove'])
        if num_insert == 1 and num_remove == 1:
            target_samples.append(sample_id)
    
    print(f"  -> Found {len(target_samples)} samples with exactly 1 insert and 1 remove")
    
    if len(target_samples) == 0:
        print("  -> No samples found matching the criteria!")
        return
    
    # 3. 随机选择10个样本
    num_to_visualize = min(10, len(target_samples))
    selected_samples = random.sample(target_samples, num_to_visualize)
    print(f"  -> Selected {num_to_visualize} samples for visualization: {selected_samples}")
    
    # 4. 获取总timestamp数
    # 通过读取sample_status_converted.npy来确定
    converted_status_file = f"{save_path}/sample_status_converted.npy"
    if os.path.exists(converted_status_file):
        converted_status = np.load(converted_status_file)
        num_timestamps = converted_status.shape[1]
    else:
        # 通过检查timestamp目录来确定
        timestamp_dirs = [d for d in os.listdir(save_path) if d.startswith('timestamp_')]
        num_timestamps = len(timestamp_dirs)
    
    print(f"  -> Total timestamps: {num_timestamps}")
    
    # 5. 为每个选中的样本收集数据并绘图
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for plot_idx, sample_id in enumerate(selected_samples):
        if plot_idx >= 10:
            break
        
        # 获取该样本的breakpoint信息
        bp_info = breakpoint_indices_dict[sample_id]
        insert_timestamp = bp_info['insert'][0] if len(bp_info['insert']) > 0 else None
        remove_timestamp = bp_info['remove'][0] if len(bp_info['remove']) > 0 else None
        
        # 收集该样本在所有timestamp下的true label probability
        timestamps = []
        true_label_probs = []
        true_label = None
        
        for k in range(num_timestamps):
            timestamp_path = f"{save_path}/timestamp_{k}/"
            
            # 读取outputs和labels
            outputs_file = f"{timestamp_path}/outputs_current.npy"
            labels_file = f"{timestamp_path}/labels.npy"
            indices_file = f"{timestamp_path}/sample_indices.npy"
            
            if not all(os.path.exists(f) for f in [outputs_file, labels_file, indices_file]):
                print(f"    -> Warning: Missing files for timestamp {k}")
                continue
            
            outputs = np.load(outputs_file)  # shape: (total_samples, num_classes)
            labels = np.load(labels_file)  # shape: (total_samples,)
            indices = np.load(indices_file)  # shape: (total_samples,)
            
            # 找到该样本在indices中的位置
            # indices保存的是样本的全局索引，顺序对应outputs和labels
            sample_pos = np.where(indices == sample_id)[0]
            if len(sample_pos) == 0:
                print(f"    -> Warning: Sample {sample_id} not found in timestamp {k}")
                continue
            
            sample_pos = sample_pos[0]
            
            # 验证索引是否在有效范围内
            if sample_pos >= len(outputs) or sample_pos >= len(labels):
                print(f"    -> Warning: Sample {sample_id} position {sample_pos} out of range in timestamp {k}")
                continue
            
            # 获取该样本的label和output
            label = int(labels[sample_pos])
            if true_label is None:
                true_label = label
            
            # 获取该样本对true label的probability
            prob = outputs[sample_pos, label]
            
            timestamps.append(k)
            true_label_probs.append(prob)
        
        # 绘图
        ax = axes[plot_idx]
        ax.plot(timestamps, true_label_probs, 'b-', linewidth=2, label='True Label Probability')
        
        # 标记insert和remove的timestamp
        if insert_timestamp is not None and insert_timestamp < len(timestamps):
            insert_idx = timestamps.index(insert_timestamp)
            ax.axvline(x=insert_timestamp, color='g', linestyle='--', linewidth=2, label='Insert')
            ax.scatter([insert_timestamp], [true_label_probs[insert_idx]], color='g', s=100, zorder=5)
        
        if remove_timestamp is not None and remove_timestamp < len(timestamps):
            remove_idx = timestamps.index(remove_timestamp)
            ax.axvline(x=remove_timestamp, color='r', linestyle='--', linewidth=2, label='Remove')
            ax.scatter([remove_timestamp], [true_label_probs[remove_idx]], color='r', s=100, zorder=5)
        
        ax.set_xlabel('Timestamp', fontsize=10)
        ax.set_ylabel('True Label Probability', fontsize=10)
        ax.set_title(f'Sample {sample_id} (Label {true_label})', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_ylim([0, 1])
    
    # 隐藏多余的子图
    for idx in range(num_to_visualize, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # 保存图片
    output_dir = os.getcwd() + f"/figures/continuous_finetune/{args['net_name']}/{args['dataset_name']}/0.01/"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/trial_{trial}_visualization.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n  -> Saved visualization to {output_file}")
    plt.close()
    
    # 6. 为每个样本单独保存一个图
    individual_output_dir = f"{output_dir}/trial_{trial}_individual/"
    os.makedirs(individual_output_dir, exist_ok=True)
    
    for sample_id in selected_samples:
        bp_info = breakpoint_indices_dict[sample_id]
        insert_timestamp = bp_info['insert'][0] if len(bp_info['insert']) > 0 else None
        remove_timestamp = bp_info['remove'][0] if len(bp_info['remove']) > 0 else None
        
        timestamps = []
        true_label_probs = []
        true_label = None
        
        for k in range(num_timestamps):
            timestamp_path = f"{save_path}/timestamp_{k}/"
            outputs_file = f"{timestamp_path}/outputs_current.npy"
            labels_file = f"{timestamp_path}/labels.npy"
            indices_file = f"{timestamp_path}/sample_indices.npy"
            
            if not all(os.path.exists(f) for f in [outputs_file, labels_file, indices_file]):
                continue
            
            outputs = np.load(outputs_file)
            labels = np.load(labels_file)
            indices = np.load(indices_file)
            
            sample_pos = np.where(indices == sample_id)[0]
            if len(sample_pos) == 0:
                continue
            
            sample_pos = sample_pos[0]
            label = int(labels[sample_pos])
            if true_label is None:
                true_label = label
            
            prob = outputs[sample_pos, label]
            timestamps.append(k)
            true_label_probs.append(prob)
        
        # 单独绘图
        plt.figure(figsize=(10, 6))
        plt.plot(timestamps, true_label_probs, 'b-', linewidth=2, label='True Label Probability', marker='o', markersize=4)
        
        if insert_timestamp is not None and insert_timestamp in timestamps:
            insert_idx = timestamps.index(insert_timestamp)
            plt.axvline(x=insert_timestamp, color='g', linestyle='--', linewidth=2, label=f'Insert (t={insert_timestamp})')
            plt.scatter([insert_timestamp], [true_label_probs[insert_idx]], color='g', s=150, zorder=5, marker='^')
        
        if remove_timestamp is not None and remove_timestamp in timestamps:
            remove_idx = timestamps.index(remove_timestamp)
            plt.axvline(x=remove_timestamp, color='r', linestyle='--', linewidth=2, label=f'Remove (t={remove_timestamp})')
            plt.scatter([remove_timestamp], [true_label_probs[remove_idx]], color='r', s=150, zorder=5, marker='v')
        
        plt.xlabel('Timestamp', fontsize=12)
        plt.ylabel('True Label Probability', fontsize=12)
        plt.title(f'Sample {sample_id} - True Label {true_label} Probability Over Time', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.ylim([0, 1])
        plt.tight_layout()
        
        individual_file = f"{individual_output_dir}/sample_{sample_id}.png"
        plt.savefig(individual_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"  -> Saved individual plots to {individual_output_dir}")

if __name__ == '__main__':
    # 示例使用
    import sys
    if len(sys.argv) > 1:
        # 从命令行参数读取
        # 格式: python visualize_finetune.py net_name dataset_name proportion trial
        net_name = sys.argv[1]
        dataset_name = sys.argv[2]
        proportion = float(sys.argv[3])
        trial = int(sys.argv[4]) if len(sys.argv) > 4 else 0
        
        args = {
            'net_name': net_name,
            'dataset_name': dataset_name,
            'proportion_of_group_unlearn': proportion
        }
        visualize_finetune_results(args, trial=trial)
    else:
        # 默认参数（需要根据实际情况修改）
        args = {
            'net_name': 'pythia70m',
            'dataset_name': 'sst5',
            'proportion_of_group_unlearn': 0.01
        }
        visualize_finetune_results(args, trial=0)

