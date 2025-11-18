from data.load_data import get_data
from data.prepare_data import split_dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import os
from torch.utils.data import Dataset, Subset, DataLoader, ConcatDataset
from model.DNN import DNN
from unlearning.utils import sample_target_samples, save_output


class WrongLabelDataset(Dataset):
    """数据集包装类，用于生成错误标签"""
    def __init__(self, dataset, num_classes):
        self.dataset = dataset
        self.num_classes = num_classes
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        if isinstance(sample, dict):
            # 文本数据
            data = {k: v for k, v in sample.items() if k != 'labels'}
            true_label = sample['labels']
            if isinstance(true_label, torch.Tensor):
                true_label = true_label.item()
            # 生成错误标签（与真实标签不同）
            wrong_label = (true_label + 1) % self.num_classes
            return {**data, 'labels': torch.tensor(wrong_label, dtype=torch.long)}
        else:
            # 图像数据
            data, true_label = sample
            if isinstance(true_label, torch.Tensor):
                true_label = true_label.item()
            # 生成错误标签（与真实标签不同）
            wrong_label = (true_label + 1) % self.num_classes
            return data, torch.tensor(wrong_label, dtype=torch.long)


def finetune_with_wrong_labels(model, forget_loader, optimizer, criterion, args, num_epochs=5):
    """使用错误标签进行finetune，达到遗忘效果"""
    model.train()
    for epoch in range(num_epochs):
        for batch in forget_loader:
            if isinstance(batch, dict):
                data = {k: v.to(args['device']) for k, v in batch.items() if k != 'labels'}
                target = batch['labels'].to(args['device'])
            else:
                data, target = batch
                data, target = data.to(args['device']), target.to(args['device'])
            
            optimizer.zero_grad()
            output = model.forward_propagation(data)
            loss = criterion(output, target)  # 使用错误标签计算损失
            loss.backward()
            optimizer.step()


def finetune_with_correct_labels(model, insert_loader, optimizer, criterion, args, num_epochs=5):
    """使用正确标签进行finetune，学习新插入的数据"""
    model.train()
    for epoch in range(num_epochs):
        for batch in insert_loader:
            if isinstance(batch, dict):
                data = {k: v.to(args['device']) for k, v in batch.items() if k != 'labels'}
                target = batch['labels'].to(args['device'])
            else:
                data, target = batch
                data, target = data.to(args['device']), target.to(args['device'])
            
            optimizer.zero_grad()
            output = model.forward_propagation(data)
            loss = criterion(output, target)  # 使用正确标签计算损失
            loss.backward()
            optimizer.step()


def continuous_update_finetune(args):
    print("dataset and net_name:", args['dataset_name'], args['net_name'])

    # 参数控制
    total_unlearn_steps = args.get('total_unlearn_steps', 80)
    update_proportion = args['proportion_of_group_unlearn']  # 0.1% 的数据更新比例
    
    print(f"  -> Total unlearn steps: {total_unlearn_steps}")
    print(f"  -> Update proportion per step: {update_proportion * 100}%")

    train_data, test_data = get_data(args['dataset_name'], args['net_name'])
    target_m, shadow_m, shadow_um = split_dataset(train_data, args['random'])

    # 获取类别数
    temp_model = DNN(args)
    num_classes = temp_model.num_classes
    del temp_model

    # 训练初始模型（基于完整target_m数据集）
    initial_loader = torch.utils.data.DataLoader(
        target_m, batch_size=args['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args['batch_size'], shuffle=False)
    
    original_model = DNN(args)
    original_model.train_model(initial_loader, test_loader)

    for t in range(args['trials']):
        print(f'\n========== The {t}-th trial ==========')
        
        # 1. 初始化索引跟踪系统
        # 合并训练集和测试集的所有样本，为每个样本分配唯一索引
        all_samples = ConcatDataset([target_m, test_data])
        total_samples = len(all_samples)
        
        # 初始化状态：target_m中的样本为1，test_data中的样本为0
        sample_status = np.zeros(total_samples, dtype=int)
        sample_status[:len(target_m)] = 1  # 训练集中的样本状态为1
        
        # 记录每个样本的原始索引映射
        # sample_status[i] = 1 表示样本i在训练集中，= 0 表示不在训练集中
        print(f"  -> Total samples: {total_samples}")
        print(f"  -> Initial training set size: {len(target_m)} (status=1)")
        print(f"  -> Initial test set size: {len(test_data)} (status=0)")
        
        # 当前训练集（初始为target_m）
        current_train_indices = list(range(len(target_m)))
        current_train_dataset = Subset(all_samples, current_train_indices)
        
        # 当前模型（从原始模型开始）
        current_model = DNN(args)
        current_model.load_state_dict(original_model.state_dict())
        
        # 保存路径
        save_path = os.getcwd() + f"/save/continuous_finetune/{args['net_name']}/{args['dataset_name']}/{args['proportion_of_group_unlearn']}/target/{t}/"
        os.makedirs(save_path, exist_ok=True)
        
        # 初始化存储结构：二维list存储每个样本在每个timestamp的状态
        # sample_status_history[i][k] 表示样本i在timestamp k的状态
        sample_status_history = []
        for i in range(total_samples):
            sample_status_history.append([sample_status[i]])  # 初始状态
        
        # 存储所有timestamp的打印信息
        timestamp_logs = []
        
        # 连续更新学习
        for k in range(total_unlearn_steps):
            print(f'\n  -> Timestamp {k}')
            
            # 初始化每个timestamp的准确率变量
            remove_acc = 0.0
            insert_acc = 0.0
            
            # 2. Remove操作：从训练集中选择0.1%的数据进行remove
            num_to_remove = max(1, int(len(current_train_indices) * update_proportion))
            if num_to_remove > 0 and len(current_train_indices) > 0:
                # 从当前训练集中随机选择要移除的样本
                remove_indices = random.sample(current_train_indices, min(num_to_remove, len(current_train_indices)))
                remove_dataset = Subset(all_samples, remove_indices)
                
                print(f"    -> Remove: {len(remove_indices)} samples from training set")

                # 使用错误标签进行finetune（遗忘效果）
                wrong_label_dataset = WrongLabelDataset(remove_dataset, num_classes)
                forget_loader = DataLoader(
                    wrong_label_dataset, 
                    batch_size=args['batch_size'], 
                    shuffle=True
                )
                optimizer = optim.Adam(current_model.parameters(), lr=args['lr'], weight_decay=5e-4)
                criterion = nn.CrossEntropyLoss()
                if args['dataset_name']=='news20':
                    remove_epoch=5
                else:
                    remove_epoch=2
                finetune_with_wrong_labels(current_model, forget_loader, optimizer, criterion, args, num_epochs=remove_epoch)
                
                # 评估remove set的准确率（使用原始正确标签）
                remove_loader_eval = DataLoader(
                    remove_dataset,
                    batch_size=args['batch_size'],
                    shuffle=False
                )
                remove_acc = current_model.test_model_acc(remove_loader_eval)

                # 更新状态：将移除的样本状态改为0
                for idx in remove_indices:
                    sample_status[idx] = 0
                
                # 从当前训练集中移除这些样本
                current_train_indices = [idx for idx in current_train_indices if idx not in remove_indices]
            
            # 3. Insert操作：从test数据集中选择0.1%的数据insert到训练集
            # 只从状态为0的样本中选择（即不在训练集中的样本）
            available_test_indices = [idx for idx in range(len(target_m), total_samples) if sample_status[idx] == 0]
            num_to_insert = max(1, int(len(current_train_indices) * update_proportion))
            
            if num_to_insert > 0 and len(available_test_indices) > 0:
                # 从可用的test样本中随机选择要插入的样本
                insert_indices = random.sample(
                    available_test_indices, 
                    min(num_to_insert, len(available_test_indices))
                )
                insert_dataset = Subset(all_samples, insert_indices)
                
                print(f"    -> Insert: {len(insert_indices)} samples from test set to training set")
                
                # 使用正确标签进行finetune（学习新数据）
                insert_loader = DataLoader(
                    insert_dataset,
                    batch_size=args['batch_size'],
                    shuffle=True
                )
                
                # Finetune with correct labels
                optimizer = optim.Adam(current_model.parameters(), lr=args['lr'], weight_decay=5e-4)
                criterion = nn.CrossEntropyLoss()
                finetune_with_correct_labels(current_model, insert_loader, optimizer, criterion, args, num_epochs=5)

                # 评估insert set的准确率
                insert_loader_eval = DataLoader(
                    insert_dataset,
                    batch_size=args['batch_size'],
                    shuffle=False
                )
                insert_acc = current_model.test_model_acc(insert_loader_eval)

                # 更新状态：将插入的样本状态改为1，并添加到训练集
                for idx in insert_indices:
                    sample_status[idx] = 1
                current_train_indices.extend(insert_indices)
            # 更新当前训练集（确保与sample_status一致）
            # 直接使用sample_status == 1来构建retain set，确保不包含unseen样本
            retain_indices = np.where(sample_status == 1)[0].tolist()
            current_train_indices = retain_indices  # 同步更新
            current_train_dataset = Subset(all_samples, retain_indices)
            
            # 在retain set上进行refine，恢复模型性能
            if len(retain_indices) > 0:
                retain_loader = DataLoader(
                    current_train_dataset,
                    batch_size=args['batch_size'],
                    shuffle=True
                )
                optimizer = optim.Adam(current_model.parameters(), lr=args['lr'], weight_decay=5e-4)
                criterion = nn.CrossEntropyLoss()
                finetune_with_correct_labels(current_model, retain_loader, optimizer, criterion, args, num_epochs=1)
                print(f"    -> Refined model on retain set ({len(retain_indices)} samples, all with status=1)")
            
            print(f"    -> Current training set size: {len(current_train_indices)}")
            print(f"    -> Samples in training set (status=1): {np.sum(sample_status == 1)}")
            print(f"    -> Samples not in training set (status=0): {np.sum(sample_status == 0)}")
            
            # 4. 更新每个样本的状态历史（记录当前timestamp的状态）
            for i in range(total_samples):
                sample_status_history[i].append(sample_status[i])
            
            # 5. 保存每个样本的output（只保存当前模型的output）
            # 为所有样本（训练集+测试集）保存output
            all_samples_loader = DataLoader(
                all_samples,
                batch_size=args['batch_size'],
                shuffle=False
            )
            
            # 保存每个样本的output
            outputs_current = []
            labels_list = []
            indices_list = []
            
            current_model.eval()
            with torch.no_grad():
                for batch_idx, batch in enumerate(all_samples_loader):
                    if isinstance(batch, dict):
                        data = {k: v.to(args['device']) for k, v in batch.items() if k != 'labels'}
                        labels = batch['labels']
                    else:
                        data, labels = batch
                        data = data.to(args['device'])
                        labels = labels
                    
                    # 只计算当前模型的输出（logits）
                    logits_current = current_model.forward_propagation(data)
                    
                    # 转换为概率
                    probs_current = F.softmax(logits_current, dim=1)
                    
                    outputs_current.append(probs_current.cpu().numpy())
                    labels_list.append(labels.numpy() if isinstance(labels, torch.Tensor) else labels)
                    
                    # 记录样本索引
                    batch_start_idx = batch_idx * args['batch_size']
                    batch_size_actual = len(labels)
                    batch_indices = list(range(batch_start_idx, batch_start_idx + batch_size_actual))
                    indices_list.extend(batch_indices)
            
            outputs_current = np.concatenate(outputs_current, axis=0)
            labels_list = np.concatenate(labels_list)
            
            # 保存到文件
            timestamp_save_path = f"{save_path}/timestamp_{k}/"
            os.makedirs(timestamp_save_path, exist_ok=True)
            
            np.save(f"{timestamp_save_path}/outputs_current.npy", outputs_current)
            np.save(f"{timestamp_save_path}/labels.npy", labels_list)
            np.save(f"{timestamp_save_path}/sample_indices.npy", np.array(indices_list))
            

            # 保存模型
            torch.save(current_model.state_dict(), f"{timestamp_save_path}/model_state_dict.pth")
            # 收集打印信息
            train_acc = current_model.test_model_acc(DataLoader(current_train_dataset, batch_size=args['batch_size'], shuffle=False))
            test_acc = current_model.test_model_acc(test_loader)
            
            log_entry = f"Timestamp {k}:\n"
            log_entry += f"  -> Training set accuracy: {train_acc:.4f}\n"
            log_entry += f"  -> Test set accuracy: {test_acc:.4f}\n"
            log_entry += f"  -> Insert set accuracy after learning: {insert_acc:.4f}\n"
            log_entry += f"  -> Remove set accuracy after unlearning: {remove_acc:.4f}\n"
            timestamp_logs.append(log_entry)
            
            print(f"    -> Training set accuracy: {train_acc:.4f}")
            print(f"    -> Test set accuracy: {test_acc:.4f}")
            print(f"    -> Insert set accuracy after learning: {insert_acc:.4f}")
            print(f"    -> Remove set accuracy after unlearning: {remove_acc:.4f}")
        
        # 6. 在trial结束后，统一保存样本状态历史
        # 转换为numpy数组：shape为 (total_samples, total_timestamps+1)
        # +1是因为包含了初始状态
        sample_status_array = np.array(sample_status_history)
        np.save(f"{save_path}/sample_status_history.npy", sample_status_array)
        print(f"\n  -> Saved sample status history to {save_path}/sample_status_history.npy")
        print(f"      Shape: {sample_status_array.shape} (samples x timestamps)")
        
        # 6.5. 转换sample_status_history为包含breakpoint信息的状态编码
        # 转换规则：
        # 0->0: 0 (连续unseen，不是breakpoint)
        # 1->1: 1 (连续retain，不是breakpoint)
        # 0->1: 2 (新增点，breakpoint)
        # 1->0: 3 (移除点，breakpoint)
        num_timestamps_with_initial = sample_status_array.shape[1]  # 包含初始状态
        num_timestamps = num_timestamps_with_initial - 1  # 排除初始状态
        
        # 初始化转换后的状态数组：shape为 (total_samples, num_timestamps)
        # 只转换timestamp 1到n（因为需要比较前一个timestamp）
        converted_status_array = np.zeros((total_samples, num_timestamps), dtype=int)
        
        for i in range(total_samples):
            for k in range(1, num_timestamps_with_initial):  # 从timestamp 1开始
                prev_status = sample_status_array[i, k-1]  # 前一个timestamp的状态
                curr_status = sample_status_array[i, k]     # 当前timestamp的状态
                
                if prev_status == 0 and curr_status == 0:
                    converted_status_array[i, k-1] = 0  # 连续unseen
                elif prev_status == 1 and curr_status == 1:
                    converted_status_array[i, k-1] = 1  # 连续retain
                elif prev_status == 0 and curr_status == 1:
                    converted_status_array[i, k-1] = 2  # 新增点（breakpoint）
                elif prev_status == 1 and curr_status == 0:
                    converted_status_array[i, k-1] = 3  # 移除点（breakpoint）
        
        # 保存转换后的状态数组
        np.save(f"{save_path}/sample_status_converted.npy", converted_status_array)
        print(f"  -> Saved converted status history to {save_path}/sample_status_converted.npy")
        print(f"      Shape: {converted_status_array.shape} (samples x timestamps)")
        print(f"      Status encoding: 0=unseen, 1=retain, 2=insert(breakpoint), 3=remove(breakpoint)")
        
        # 统计转换后的状态分布（用于后续统计）
        converted_stats = {
            0: np.sum(converted_status_array == 0),
            1: np.sum(converted_status_array == 1),
            2: np.sum(converted_status_array == 2),
            3: np.sum(converted_status_array == 3)
        }
        # 7. 基于转换后的状态进行详细统计（主要统计）
        num_unseen_per_sample = np.sum(converted_status_array == 0, axis=1)  # 每个样本的unseen个数
        num_retain_per_sample = np.sum(converted_status_array == 1, axis=1)  # 每个样本的retain个数
        num_insert_per_sample = np.sum(converted_status_array == 2, axis=1)  # 每个样本的insert个数
        num_remove_per_sample = np.sum(converted_status_array == 3, axis=1)  # 每个样本的remove个数
        num_breakpoints_per_sample = num_insert_per_sample + num_remove_per_sample
        
        # 统计详细模式
        # 1. No breakpoint情况：区分全部unseen和全部retain
        all_unseen_samples = np.where((num_breakpoints_per_sample == 0) & (num_unseen_per_sample == num_timestamps))[0]
        all_retain_samples = np.where((num_breakpoints_per_sample == 0) & (num_retain_per_sample == num_timestamps))[0]
        
        # 2. 有breakpoint情况：统计insert和remove的数量分布
        has_breakpoint_samples = np.where(num_breakpoints_per_sample > 0)[0]
        
        # 统计各种模式的数量
        pattern_statistics = {
            'all_unseen': len(all_unseen_samples),
            'all_retain': len(all_retain_samples),
        }
        
        # 对于有breakpoint的样本，按insert和remove的数量分类
        for num_insert in range(num_timestamps + 1):
            for num_remove in range(num_timestamps + 1):
                if num_insert + num_remove == 0:
                    continue  # 跳过no breakpoint的情况
                if num_insert + num_remove > num_timestamps:
                    continue
                mask = (num_insert_per_sample == num_insert) & (num_remove_per_sample == num_remove)
                count = np.sum(mask)
                if count > 0:
                    pattern_key = f"{num_insert}_insert_{num_remove}_remove"
                    pattern_statistics[pattern_key] = count
        
        # 保存breakpoint索引
        # 对于每个样本，记录breakpoint的timestamp和类型
        breakpoint_indices_dict = {}
        for i in range(total_samples):
            insert_indices = np.where(converted_status_array[i, :] == 2)[0].tolist()  # insert breakpoint的timestamp
            remove_indices = np.where(converted_status_array[i, :] == 3)[0].tolist()  # remove breakpoint的timestamp
            if len(insert_indices) > 0 or len(remove_indices) > 0:
                breakpoint_indices_dict[i] = {
                    'insert': insert_indices,
                    'remove': remove_indices
                }
        
        # 保存breakpoint索引到文件
        breakpoint_indices_file = f"{save_path}/breakpoint_indices.npy"
        # 使用字典保存，因为每个样本的breakpoint数量不同
        np.save(breakpoint_indices_file, breakpoint_indices_dict, allow_pickle=True)
        print(f"  -> Saved breakpoint indices to {breakpoint_indices_file}")
        
        # 打印统计结果
        print(f"\n  -> Converted Status Pattern Statistics (across {num_timestamps} timestamps):")
        print(f"      {'Pattern':<40} {'Count':<15} {'Percentage':<15}")
        print(f"      {'-' * 70}")
        
        # No breakpoint情况
        all_unseen_count = pattern_statistics['all_unseen']
        all_retain_count = pattern_statistics['all_retain']
        all_unseen_pct = (all_unseen_count / total_samples) * 100 if total_samples > 0 else 0.0
        all_retain_pct = (all_retain_count / total_samples) * 100 if total_samples > 0 else 0.0
        print(f"      {'All Unseen (no breakpoint)':<40} {all_unseen_count:<15} {all_unseen_pct:>13.2f}%")
        print(f"      {'All Retain (no breakpoint)':<40} {all_retain_count:<15} {all_retain_pct:>13.2f}%")
        
        # 有breakpoint情况
        for pattern_key in sorted(pattern_statistics.keys()):
            if pattern_key in ['all_unseen', 'all_retain']:
                continue
            count = pattern_statistics[pattern_key]
            percentage = (count / total_samples) * 100 if total_samples > 0 else 0.0
            # 解析pattern_key: "num_insert_insert_num_remove_remove"
            parts = pattern_key.split('_')
            num_insert = int(parts[0])
            num_remove = int(parts[2])
            pattern_desc = f"{num_insert} insert, {num_remove} remove breakpoints"
            print(f"      {pattern_desc:<40} {count:<15} {percentage:>13.2f}%")
        
        # 保存主要统计结果到文件（基于转换后的状态）
        main_stats_file_path = f"{save_path}/status_statistics.txt"
        with open(main_stats_file_path, 'w', encoding='utf-8') as f:
            f.write(f"Trial {t} - Converted Status Pattern Statistics\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Total samples: {total_samples}\n")
            f.write(f"Total timestamps: {num_timestamps}\n\n")
            f.write(f"Status encoding:\n")
            f.write(f"  0: Unseen (0->0, not breakpoint)\n")
            f.write(f"  1: Retain (1->1, not breakpoint)\n")
            f.write(f"  2: Insert (0->1, breakpoint)\n")
            f.write(f"  3: Remove (1->0, breakpoint)\n\n")
            f.write(f"Pattern Distribution:\n")
            f.write(f"{'Pattern':<50} {'Count':<15} {'Percentage':<15}\n")
            f.write("-" * 80 + "\n")
            
            # No breakpoint情况
            f.write(f"{'All Unseen (no breakpoint)':<50} {all_unseen_count:<15} {all_unseen_pct:>13.2f}%\n")
            f.write(f"{'All Retain (no breakpoint)':<50} {all_retain_count:<15} {all_retain_pct:>13.2f}%\n")
            
            # 有breakpoint情况
            for pattern_key in sorted(pattern_statistics.keys()):
                if pattern_key in ['all_unseen', 'all_retain']:
                    continue
                count = pattern_statistics[pattern_key]
                percentage = (count / total_samples) * 100 if total_samples > 0 else 0.0
                parts = pattern_key.split('_')
                num_insert = int(parts[0])
                num_remove = int(parts[2])
                pattern_desc = f"{num_insert} insert, {num_remove} remove breakpoints"
                f.write(f"{pattern_desc:<50} {count:<15} {percentage:>13.2f}%\n")
            
            # 添加详细的状态统计
            f.write(f"\n\nDetailed Status Count Statistics:\n")
            f.write(f"{'Status Type':<30} {'Total Count':<20} {'Average per Sample':<20}\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Unseen (0->0)':<30} {converted_stats[0]:<20} {converted_stats[0]/total_samples:.2f}\n")
            f.write(f"{'Retain (1->1)':<30} {converted_stats[1]:<20} {converted_stats[1]/total_samples:.2f}\n")
            f.write(f"{'Insert (0->1)':<30} {converted_stats[2]:<20} {converted_stats[2]/total_samples:.2f}\n")
            f.write(f"{'Remove (1->0)':<30} {converted_stats[3]:<20} {converted_stats[3]/total_samples:.2f}\n")
        print(f"  -> Saved main status statistics to {main_stats_file_path}")
        
        # 8. 保存所有timestamp的打印信息
        log_file_path = f"{save_path}/timestamp_logs.txt"
        with open(log_file_path, 'w', encoding='utf-8') as f:
            f.write(f"Trial {t} - Timestamp Logs\n")
            f.write("=" * 50 + "\n\n")
            for log_entry in timestamp_logs:
                f.write(log_entry + "\n")
        print(f"  -> Saved timestamp logs to {log_file_path}")
