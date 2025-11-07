from data.load_data import get_data
from data.prepare_data import split_dataset
import torch
import torch.nn as nn
import torch.optim as optim
from model.DNN import DNN
from unlearning.utils import sample_target_samples, save_output, l1_regularization
from torch.utils.data import ConcatDataset
import copy
import time
import numpy as np
import torch.nn.functional as F

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

            # 使用GA进行遗忘学习

            current_model = scrub_unlearn(current_model, forget_loader, retain_loader, test_loader, args)
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




class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network using KL Divergence."""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='sum') * (self.T ** 2) / y_s.shape[0]
        return loss




def scrub_unlearn(original_model,forget_loader,retain_loader,test_loader,args):
        device=args['device']
        T=1
        scrub_beta=0.0
        scrub_gamma=0.99
        scrub_alpha = 0.1
        smoothing=0.0
        m_steps = 1
        unlearn_epochs=30
        lr=0.001

        teacher = copy.deepcopy(original_model)
        student = copy.deepcopy(original_model)
        model_t = copy.deepcopy(teacher)
        model_s = copy.deepcopy(student)

        module_list = nn.ModuleList([])
        module_list.append(model_s)
        trainable_list = nn.ModuleList([])
        trainable_list.append(model_s)

        criterion_cls = nn.CrossEntropyLoss()
        criterion_div = DistillKL(T)
        criterion_kd = DistillKL(T)

        criterion_list = nn.ModuleList([])
        criterion_list.append(criterion_cls)  # classification loss
        criterion_list.append(criterion_div)  # KL divergence loss, original knowledge distillation
        criterion_list.append(criterion_kd)  # other knowledge distillation loss

        # optimizer
        optimizer = optim.Adam(trainable_list.parameters(),
                              lr=lr,weight_decay=5e-4)

        module_list.append(model_t)

        if torch.cuda.is_available():
            module_list.to(device)
            criterion_list.to(device)
            import torch.backends.cudnn as cudnn
            cudnn.benchmark = True


        def avg_fn(averaged_model_parameter, model_parameter, num_averaged):
            return (1 - scrub_beta) * averaged_model_parameter + scrub_beta* model_parameter

        swa_model = torch.optim.swa_utils.AveragedModel(model_s, avg_fn=avg_fn)
        swa_model.to(device)

        for epoch in range(1,unlearn_epochs + 1):
          #  lr_decay_epochs = [1, 2, 3]
          #  lr_decay_rate=0.1
          #  lr, optimizer = adjust_learning_rate(epoch, lr, lr_decay_epochs, lr_decay_rate, optimizer)
            maximize_loss = 0
            if epoch <= m_steps:
                maximize_loss = train_distill(forget_loader, module_list, swa_model,
                                              criterion_list, optimizer, scrub_gamma,scrub_alpha,scrub_beta, smoothing,"maximize",device,
                                              quiet=False)
            train_acc, train_loss = train_distill(retain_loader, module_list, swa_model, criterion_list,
                                                  optimizer, scrub_gamma,scrub_alpha,scrub_beta, smoothing,
                                                  "minimize", device,quiet=False)

          #  print("maximize loss: {:.2f}\t minimize loss: {:.2f}\t train_acc: {}".format(maximize_loss,
            #                                                                              train_loss, train_acc))
            # retain_set_acc = model_s.test_model_acc(retain_loader)
            # forget_set_acc = model_s.test_model_acc(forget_loader)
            # test_acc = model_s.test_model_acc(test_loader)
            # print(f'forget set acc (UA) %s  | retain set acc (RA) %s |test acc (TA) %s ' % (
            # round(forget_set_acc, 4), round(retain_set_acc, 4), round(test_acc, 4)))

        return model_s


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def param_dist(model, swa_model, p):
    #This is from https://github.com/ojus1/SmoothedGradientDescentAscent/blob/main/SGDA.py
    dist = 0.
    for p1, p2 in zip(model.parameters(), swa_model.parameters()):
        dist += torch.norm(p1 - p2, p='fro')
    return p * dist


def train_distill(train_loader, module_list, swa_model, criterion_list, optimizer, scrub_gamma,scrub_alpha,scrub_beta,smoothing, split,device,quiet=False):
    """One epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    kd_losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    loss = 0.0

    for idx, data in enumerate(train_loader):
        input, target = data
        data_time.update(time.time() - end)
        input = input.float()
        if torch.cuda.is_available():
            input = input.to(device)
            target = target.to(device)


        # ===================forward=====================
        #feat_s, logit_s = model_s(input, is_feat=True, preact=False)
        logit_s = model_s(input)
        with torch.no_grad():
            logit_t = model_t(input)

        loss_cls = criterion_cls(logit_s, target)
        loss_div = criterion_div(logit_s, logit_t)
        loss_kd = 0
        if split == "minimize":
            loss = scrub_gamma * loss_cls + scrub_alpha * loss_div +scrub_beta * loss_kd
        elif split == "maximize":
            loss = -loss_div

        loss = loss + param_dist(model_s, swa_model, smoothing)

        if split == "minimize" and not quiet:
            acc1, _ = accuracy(logit_s, target, topk=(1, 1))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
        elif split == "maximize" and not quiet:
            kd_losses.update(loss.item(), input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()

        loss.backward()
        # nn.utils.clip_grad_value_(model_s.parameters(), clip)
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

    if split == "minimize":
        # if not quiet:
        #     print(' * Acc@1 {top1.avg:.3f} '
        #           .format(top1=top1))

        return top1.avg, losses.avg
    else:
        return kd_losses.avg


def adjust_learning_rate(epoch, learning_rate, lr_decay_epochs, lr_decay_rate, optimizer):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    steps = np.sum(epoch > np.asarray(lr_decay_epochs))
    new_lr = learning_rate
    if steps > 0:
        new_lr = learning_rate * (lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    return new_lr, optimizer