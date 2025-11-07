from data.load_data import get_data
from data.prepare_data import split_dataset
import torch
import torch.nn as nn
import torch.optim as optim
from model.DNN import DNN
from unlearning.utils import sample_target_samples, save_output
from torch.utils.data import ConcatDataset


def continuous_unlearn_NPO(args):
    print("dataset and net_name:", args['dataset_name'], args['net_name'])

    # 参数控制
    total_unlearn_steps = args.get('total_unlearn_steps', 50)  # 总遗忘轮数
    add_data_A_step = args.get('add_data_A_step', 19)  # 加入数据A的轮次（0-based）
    forget_data_A_step = args.get('forget_data_A_step', 29)  # 遗忘数据A的轮次（0-based）
    
    print(f"  -> Total unlearn steps: {total_unlearn_steps}")
    print(f"  -> Add data A at step: {add_data_A_step}")
    print(f"  -> Forget data A at step: {forget_data_A_step}")

    train_data, test_data = get_data(args['dataset_name'], args['net_name'])

    target_m, shadow_m, shadow_um = split_dataset(train_data, args['random'])

    def collate_fn(batch):
        if isinstance(batch[0], dict):
            input_ids = torch.stack([item['input_ids'] for item in batch])
            attention_mask = torch.stack([item['attention_mask'] for item in batch])
            labels = torch.stack([item['labels'] for item in batch])
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }
        else:
            return torch.utils.data.dataloader.default_collate(batch)

    train_loader = torch.utils.data.DataLoader(
        target_m, batch_size=args['batch_size'], shuffle=True, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args['batch_size'], shuffle=False, collate_fn=collate_fn)

    original_model = DNN(args)
    # original_model.train_model(train_loader, test_loader)

    for t in range(args['trials']):
        print(f'The {t}-th trials')

        # 从原始target_m中固定数据A并移除
        data_A, remaining_data = sample_target_samples(
            target_m, 
            args['proportion_of_group_unlearn'], 
            args['dataset_name'], 
            False
        )
        print(f"  -> Fixed data A size: {len(data_A)}, remaining data size: {len(remaining_data)}")
        
        # 从移除A后的数据集开始（这是第0轮的起始数据集）
        current_dataset = remaining_data
        
        # 训练初始模型（基于完整数据集）
        initial_loader = torch.utils.data.DataLoader(
            target_m, batch_size=args['batch_size'], shuffle=True, collate_fn=collate_fn)
        original_model = DNN(args)
        original_model.train_model(initial_loader, test_loader)
        
        # 初始化当前模型为原始模型
        current_model = DNN(args)
        current_model.load_state_dict(original_model.state_dict())
        
        # 用于保存各轮模型的快照
        model_history = []
        
        # 连续遗忘学习
        for k in range(total_unlearn_steps):
            print(f"  -> Continuous unlearn step {k}")
            
            if k == add_data_A_step:  # 加入数据A的轮次
                print(f"    -> Step {k}: Adding data A + normal unlearning")
                # 1) 正常遗忘：更新到 retain_set
                forget_set, retain_set = sample_target_samples(
                    current_dataset,
                    args['proportion_of_group_unlearn'],
                    args['dataset_name'],
                    False
                )
                # 2) 加入数据A：作为本轮最终训练数据集
                current_dataset = ConcatDataset([retain_set, data_A])
                
            elif k == forget_data_A_step:  # 遗忘数据A的轮次
                print(f"    -> Step {k}: Forgetting data A")
                # 从当前数据集中移除A部分
                if isinstance(current_dataset, ConcatDataset):
                    # 如果current_dataset是ConcatDataset，只保留第一部分（非A数据）
                    current_dataset = current_dataset.datasets[0]
                else:
                    # 如果current_dataset不包含A，说明A已经被移除了，直接使用
                    pass

            elif add_data_A_step < k < forget_data_A_step:  # 加入A后到遗忘A前的轮次：不能遗忘数据A
                print(f"    -> Step {k}: Unlearning (cannot forget data A)")
                # 从当前数据集中采样，但确保数据A始终在retain_set中
                if isinstance(current_dataset, ConcatDataset):
                    # 如果current_dataset是ConcatDataset，只从第一部分（非A数据）采样
                    non_A_dataset = current_dataset.datasets[0]
                    data_A = current_dataset.datasets[1]  # 获取数据A
                else:
                    # 如果current_dataset不包含A，直接使用
                    non_A_dataset = current_dataset
                
                # 从非A数据集中采样遗忘子集
                forget_set, retain_set = sample_target_samples(
                    non_A_dataset,
                    args['proportion_of_group_unlearn'],
                    args['dataset_name'],
                    False
                )
                # 确保数据A始终在retain_set中，作为本轮最终训练数据集
                current_dataset = ConcatDataset([retain_set, data_A])
                
            else:  # 其他轮次：正常遗忘学习
                print(f"    -> Step {k}: Normal unlearning")
                # 基于上一次的训练数据集随机采样数据然后剔除后进行重新学习
                forget_set, retain_set = sample_target_samples(
                    current_dataset,
                    args['proportion_of_group_unlearn'],
                    args['dataset_name'],
                    False
                )
                current_dataset = retain_set
            
            # 统一在每一轮末尾执行NPO遗忘学习
            # 基于上一轮的模型进行连续遗忘学习
            prev_model = current_model  # 保存上一轮的模型作为参考
            current_model = DNN(args)
            current_model.load_state_dict(prev_model.state_dict())  # 从上一轮模型开始
            
            # 准备数据加载器
            forget_loader = torch.utils.data.DataLoader(
                forget_set, batch_size=args['batch_size'], shuffle=False, collate_fn=collate_fn)
            retain_loader = torch.utils.data.DataLoader(
                current_dataset, batch_size=args['batch_size'], shuffle=False, collate_fn=collate_fn)
            
            # 使用NPO进行遗忘学习（基于上一轮模型）
            current_model = NPO_train(prev_model, current_model, forget_loader, retain_loader, test_loader, args)
            
            # 记录模型历史
            model_history.append((k, current_model))
            
            # 计算并打印数据A在当前模型下的预测准确率
            data_A_loader = torch.utils.data.DataLoader(
                data_A, batch_size=args['batch_size'], shuffle=False, collate_fn=collate_fn)
            data_A_acc = current_model.test_model_acc(data_A_loader)
            print(f"    -> Data A accuracy at step {k}: {data_A_acc:.4f}")
            
            # 保存数据A在当前模型上的输出
            save_output(
                f'tracked_data_A',
                args,
                original_model,
                current_model,
                data_A,  # 始终跟踪数据A
                current_dataset,  # 使用A作为retain_set（仅用于保存格式）
                test_data,
                shadow_um,
                t,
                k
            )


def l2_penalty(model, model_init, weight_decay):
    l2_loss = 0
    for (k, p), (k_init, p_init) in zip(model.named_parameters(), model_init.named_parameters()):
        if p.requires_grad:
            l2_loss += (p - p_init).pow(2).sum()
    l2_loss *= (weight_decay / 2.)
    return l2_loss


def NPO_train(original_model, unlearned_model, forget_loader, retain_loader, test_loader, args):
    unlearn_epoch = 10
    lr_npo=args['lr']
    if args['dataset_name'] == 'sst5':
        lr_npo=args['lr']
    if args['dataset_name'] == 'news20':
        lr_npo=args['lr'] * 3.0
    if args['dataset_name'] == 'rte':
        lr_npo=args['lr']
    if args['dataset_name'] == 'mrpc':
        lr_npo=args['lr']
    optimizer_NPO = optim.Adam(unlearned_model.parameters(), lr=lr_npo , weight_decay=5e-4)
    optimizer_remained = optim.Adam(unlearned_model.parameters(), lr=args['lr'], weight_decay=5e-4)

    criterion = nn.CrossEntropyLoss()
    unlearned_model.train()

    for t in range(unlearn_epoch):
        NPO_unlearn(unlearned_model, forget_loader, optimizer_NPO, original_model, args)
        retain_set_acc = unlearned_model.test_model_acc(retain_loader)
        forget_set_acc = unlearned_model.test_model_acc(forget_loader)
        test_acc = unlearned_model.test_model_acc(test_loader)
        print('epoch %s: forget set acc (UA) %s  | retain set acc (RA) %s |test acc (TA) %s ' % (
        t, round(forget_set_acc, 4), round(retain_set_acc, 4), round(test_acc, 4)))

        refine_remained(unlearned_model, retain_loader, optimizer_remained, criterion, args)
        retain_set_acc = unlearned_model.test_model_acc(retain_loader)
        forget_set_acc = unlearned_model.test_model_acc(forget_loader)
        test_acc = unlearned_model.test_model_acc(test_loader)
        print('epoch %s: forget set acc (UA) %s  | retain set acc (RA) %s |test acc (TA) %s ' % (
        t, round(forget_set_acc, 4), round(retain_set_acc, 4), round(test_acc, 4)))

    return unlearned_model

def NPO_unlearn(unlearned_model, forget_loader, optimizer, original_model, args):
    beta=0.1
    for forget_batch in forget_loader:
        if isinstance(forget_batch, dict):
            inputs_f = {k: v.to(args['device']) for k, v in forget_batch.items() if k != 'labels'}
            targets_f = forget_batch['labels'].to(args['device'])
        else:
            inputs_f, targets_f = forget_batch
            inputs_f, targets_f = inputs_f.to(args['device']), targets_f.to(args['device'])

        optimizer.zero_grad()

        with torch.no_grad():
            outputs_target = original_model.forward_propagation(inputs_f)

        outputs_theta = unlearned_model.forward_propagation(inputs_f)


        probs_theta = torch.softmax(outputs_theta, dim=1)
        probs_target = torch.softmax(outputs_target, dim=1)

        target_probs_theta = torch.gather(probs_theta, 1, targets_f.unsqueeze(1)).squeeze(1)
        target_probs_target = torch.gather(probs_target, 1, targets_f.unsqueeze(1)).squeeze(1)

        epsilon = 1e-8
        log_ratio = torch.log(target_probs_theta + epsilon) - torch.log(target_probs_target + epsilon)

        npo_loss = -(2.0 / beta) * torch.log(torch.sigmoid(-beta * log_ratio) + epsilon)
        npo_loss = npo_loss.mean()


        npo_loss.backward()
        optimizer.step()

def refine_remained(unlearned_model, retain_loader, optimizer, criterion, args):

    for batch in retain_loader:
        if isinstance(batch, dict):
            data = {k: v.to(args['device']) for k, v in batch.items() if k != 'labels'}
            target = batch['labels'].to(args['device'])
        else:
            data, target = batch
            data, target = data.to(args['device']), target.to(args['device'])

        optimizer.zero_grad()
        output = unlearned_model.forward_propagation(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()