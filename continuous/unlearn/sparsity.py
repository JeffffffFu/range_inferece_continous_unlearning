from data.load_data import get_data
from data.prepare_data import split_dataset
import torch
import torch.nn as nn
import torch.optim as optim
from model.DNN import DNN
from unlearning.utils import sample_target_samples, save_output, l1_regularization
from torch.utils.data import ConcatDataset


def continuous_unlearn_sparsity(args):
    print("dataset and net_name:", args['dataset_name'], args['net_name'])

    # 参数控制
    total_unlearn_steps = args.get('total_unlearn_steps', 50)  # 总遗忘轮数
    add_data_A_step = args.get('add_data_A_step', 19)  # 加入数据A的轮次（0-based）
    forget_data_A_step = args.get('forget_data_A_step', 29)  # 遗忘数据A的轮次（0-based）

    print(f"  -> Total unlearn steps: {total_unlearn_steps}")
    print(f"  -> Add data A at step: {add_data_A_step}")
    print(f"  -> Forget data A at step: {forget_data_A_step}")

    train_data, test_data = get_data(args['dataset_name'])

    target_m, shadow_m, shadow_um = split_dataset(train_data, args['random'])
    target_m = train_data

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

        # 在移除数据A后的数据集上训练初始模型
        initial_loader = torch.utils.data.DataLoader(
            remaining_data, batch_size=args['batch_size'], shuffle=True, collate_fn=collate_fn)
        current_model = DNN(args)
        current_model.train_model(initial_loader, test_loader)

        # 从移除A后的数据集开始
        current_dataset = remaining_data

        # 用于保存各轮模型的快照
        model_history = []

        # 连续遗忘学习
        for k in range(total_unlearn_steps):
            print(f"  -> Continuous unlearn step {k}")

            if k == add_data_A_step :  # 加入数据A的轮次
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
                    forget_set=data_A
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

            # 统一在每一轮末尾执行GA遗忘学习
            # 每轮都从原始模型开始进行遗忘学习（符合连续遗忘逻辑）
            # 注意：这里每轮都从原始模型开始，这是正确的连续遗忘逻辑
            # 因为每轮遗忘的是不同的数据子集，需要基于原始模型进行遗忘

            # 准备数据加载器
            forget_loader = torch.utils.data.DataLoader(
                forget_set, batch_size=args['batch_size'], shuffle=False, collate_fn=collate_fn)
            retain_loader = torch.utils.data.DataLoader(
                current_dataset, batch_size=args['batch_size'], shuffle=False, collate_fn=collate_fn)


            current_model = sparsity_train(current_model, retain_loader, forget_loader, test_loader, args)
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


def sparsity_train(unlearned_model,retain_loader,forget_loader,test_loader,args,with_l1=False):

   # optimizer = optim.SGD(unlearned_model.parameters(), 0.001, momentum=0.9)
    optimizer = optim.Adam(unlearned_model.parameters(), 0.0005, weight_decay=5e-4)

    criterion = nn.CrossEntropyLoss()

    num_epochs = 20
    no_l1_epochs=0
    sparse_scheduler= "increase"
    sno_l1_epochs=0
    alpha=0.0005
    unlearned_model.train()

    for epoch in range(num_epochs):
        losses = []
        current_alpha=0.
        if epoch <= num_epochs - no_l1_epochs:
            if sparse_scheduler == 'decay':
                current_alpha = (2 - (2 * epoch / num_epochs)) * alpha
            elif sparse_scheduler == 'constant':
                current_alpha = alpha
            elif sparse_scheduler == 'increase':
                current_alpha = (2 * epoch / num_epochs) * alpha
        elif epoch > num_epochs - sno_l1_epochs:
            current_alpha = 0

        for data, target  in retain_loader:

            data, target = data.to(args['device']), target.to(args['device'])
            optimizer.zero_grad()
            output = unlearned_model.forward_propagation(data)
            loss = criterion(output, target)
            if with_l1:
                loss = loss + current_alpha * l1_regularization(unlearned_model)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        #
        # retain_set_acc =unlearned_model.test_model_acc(retain_loader)
        # forget_set_acc =unlearned_model.test_model_acc(forget_loader)
        # test_acc = unlearned_model.test_model_acc(test_loader)
        # print('epoch %s: forget set acc (UA) %s  | retain set acc (RA) %s |test acc (TA) %s ' % (epoch,  round(forget_set_acc, 4), round(retain_set_acc, 4),  round(test_acc, 4)))
        #
    return unlearned_model