import os
import random
import time
import copy

import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from attack.Double_Attack import attack_feature_base, MLP2Layer
from attack.utils import baseline_prep_for_double_attack
from data.load_data import get_data
from data.prepare_data import construct_dataset, split_dataset, split_dataset2, split_dataset3
from model.DNN import DNN
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.utils.data import ConcatDataset
from torchvision import datasets, transforms
from scipy.stats import pearsonr

from parameter_parser import parameter_parser
from unlearning.utils import sample_target_samples, save_output, calculate_confidence_with_subsets, TransformedDataset, \
    sample_target_samples2, get_top_bottom_n_indices, add_gaussian_noise, l1_regularization, pruning_model, check_sparsity, extract_mask, \
    remove_prune, prune_model_custom


# just save posterior


# need to motify main.py, and let --U_method all at .sh
def retain_eva(args):
    print("retain_eva----dataset and net_name:",args['dataset_name'],args['net_name'])

    train_data, test_data = get_data(args['dataset_name'], model_name=args.get('net_name'))

    target_m,shadow_m,shadow_um = split_dataset(train_data, args['random'])

    train_loader = torch.utils.data.DataLoader(
        target_m, batch_size=args['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args['batch_size'], shuffle=True)
    args['num_epochs']=25
    
    # 训练一个original_model，所有unlearn方法共享
    original_model = DNN(args)
    original_model.train_model(train_loader, test_loader)
    acc_train_loader = original_model.test_model_acc(train_loader)
    acc_test_loader = original_model.test_model_acc(test_loader)
    print(f"Original model - Train acc: {acc_train_loader:.4f}, Test acc: {acc_test_loader:.4f}")

    # 定义要测试的unlearn方法列表
    unlearn_methods = ['retrain', 'sparsity', 'scrub', 'GA']
    # 如果args中指定了U_method，则只测试该方法
    if 'U_method' in args and args['U_method'] != 'all':
        unlearn_methods = [args['U_method']]
    
    # 为sparsity方法准备剪枝模型（提前准备，避免重复计算）
    initial_model = None
    pruned_model = None
    if 'sparsity' in unlearn_methods:
        initial_model = copy.deepcopy(original_model)
        # 对原始模型进行剪枝，剪枝比例可根据需要调整
        if args['dataset_name']=='cifar10' or args['dataset_name']=='cifar100':
            prune = 0.8
        else:
            prune=0.00001
        pruned_original_model = pruning_model(copy.deepcopy(original_model), prune)
        remain_weight = check_sparsity(pruned_original_model)
        # 提取剪枝后的mask
        current_mask = extract_mask(pruned_original_model.state_dict())
        remove_prune(pruned_original_model)
        pruned_model = prune_model_custom(pruned_original_model, current_mask, args)
        print(f"Sparsity pruning completed. Remaining weight ratio: {remain_weight:.2f}%")

    # 对每个trial，使用相同的forget_set和retain_set，测试不同的unlearn方法
    for t in range(args['trials']):
        print(f'\n========== The {t}-th trial ==========')
        
        # 每个trial采样一次forget_set和retain_set
        forget_set, retain_set = sample_target_samples(target_m, args['proportion_of_group_unlearn'],args['dataset_name'],False)
        retain_loader = torch.utils.data.DataLoader(
            retain_set, batch_size=args['batch_size'], shuffle=True)
        forget_loader = torch.utils.data.DataLoader(
            forget_set, batch_size=args['batch_size'], shuffle=False)
        
        # 循环不同的unlearn方法
        for method in unlearn_methods:
            print(f'\n--- Testing {method} method ---')
            
            # 临时修改U_method用于保存路径
            original_u_method = args.get('U_method', 'all')
            args['U_method'] = method
            
            try:
                if method == 'retrain':
                    args['num_epochs'] = 40
                    unlearned_model = DNN(args)
                    unlearned_model.train_model(retain_loader, test_loader)
                    model_for_save = original_model
                    
                elif method == 'sparsity':
                    # 用sparsity的unlearn方法
                    unlearned_model = DNN(args)
                    unlearned_model.load_state_dict(pruned_model.state_dict(), strict=False)
                    unlearned_model = sparsity_train(unlearned_model, retain_loader, forget_loader, test_loader, args)
                    model_for_save = initial_model
                    
                elif method == 'scrub':
                    # 用scrub的unlearn方法
                    unlearned_model = DNN(args)
                    unlearned_model.load_state_dict(original_model.state_dict())
                    unlearned_model = scrub_unlearn(unlearned_model, forget_loader, retain_loader, test_loader, args['device'])
                    model_for_save = original_model
                    
                elif method == 'GA':
                    # 用GA的unlearn方法
                    unlearned_model = DNN(args)
                    unlearned_model.load_state_dict(original_model.state_dict())
                    unlearned_model = GA_train(unlearned_model, forget_loader, retain_loader, test_loader, args)
                    model_for_save = original_model
                    
                else:
                    print(f"Warning: Unknown unlearning method: {method}, skipping...")
                    continue
                
                # 保存结果
                save_output('target', args, model_for_save, unlearned_model, forget_set, retain_set, test_data, shadow_um, t)
                print(f"✓ {method} method completed and saved")
                
            except Exception as e:
                print(f"✗ Error in {method} method: {e}")
                import traceback
                traceback.print_exc()
            finally:
                # 恢复原始的U_method
                args['U_method'] = original_u_method


# ==================== Sparsity Unlearning Methods ====================

def sparsity_train(unlearned_model, retain_loader, forget_loader, test_loader, args, with_l1=False):
    """
    使用sparsity方法进行unlearning训练
    可以调整的参数：
    - num_epochs: 训练轮数 (默认: 20)
    - lr: 学习率 (默认: 0.0005)
    - weight_decay: 权重衰减 (默认: 5e-4)
    - no_l1_epochs: 不使用L1正则化的轮数 (默认: 0)
    - sparse_scheduler: 稀疏调度器类型 'decay', 'constant', 'increase' (默认: "increase")
    - sno_l1_epochs: 最后不使用L1的轮数 (默认: 0)
    - alpha: L1正则化系数 (默认: 0.0005)
    """
    # 可调整的参数
    optimizer = optim.Adam(unlearned_model.parameters(), 0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    num_epochs = 35
    no_l1_epochs = 0
    sparse_scheduler = "increase"
    sno_l1_epochs = 0
    alpha = 0.0005
    unlearned_model.train()

    for epoch in range(num_epochs):
        current_alpha = 0.
        if epoch <= num_epochs - no_l1_epochs:
            if sparse_scheduler == 'decay':
                current_alpha = (2 - (2 * epoch / num_epochs)) * alpha
            elif sparse_scheduler == 'constant':
                current_alpha = alpha
            elif sparse_scheduler == 'increase':
                current_alpha = (2 * epoch / num_epochs) * alpha
        elif epoch > num_epochs - sno_l1_epochs:
            current_alpha = 0

        for data, target in retain_loader:
            data, target = data.to(args['device']), target.to(args['device'])
            optimizer.zero_grad()
            output = unlearned_model.forward_propagation(data)
            loss = criterion(output, target)
            if with_l1:
                loss = loss + current_alpha * l1_regularization(unlearned_model)
            loss.backward()
            optimizer.step()

        retain_set_acc =unlearned_model.test_model_acc(retain_loader)
        # forget_set_acc =unlearned_model.test_model_acc(forget_loader)
        # test_acc = unlearned_model.test_model_acc(test_loader)
        print('epoch %s: retain set acc (RA) %s ' % (epoch, round(retain_set_acc, 4)))



    return unlearned_model


# ==================== SCRUB Unlearning Methods ====================

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
    """计算模型参数与SWA模型之间的距离"""
    dist = 0.
    for p1, p2 in zip(model.parameters(), swa_model.parameters()):
        dist += torch.norm(p1 - p2, p='fro')
    return p * dist


def train_distill(train_loader, module_list, swa_model, criterion_list, optimizer, scrub_gamma, scrub_alpha, scrub_beta, smoothing, split, device, quiet=False):
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
        logit_s = model_s(input)
        with torch.no_grad():
            logit_t = model_t(input)

        loss_cls = criterion_cls(logit_s, target)
        loss_div = criterion_div(logit_s, logit_t)
        loss_kd = 0
        if split == "minimize":
            loss = scrub_gamma * loss_cls + scrub_alpha * loss_div + scrub_beta * loss_kd
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
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

    if split == "minimize":
        return top1.avg, losses.avg
    else:
        return kd_losses.avg


def scrub_unlearn(original_model, forget_loader, retain_loader, test_loader, device):
    """
    使用SCRUB方法进行unlearning
    可以调整的参数：
    - T: 温度参数 (默认: 1)
    - scrub_beta: SWA平均化参数 (默认: 0.0)
    - scrub_gamma: 分类损失权重 (默认: 0.99)
    - scrub_alpha: KL散度损失权重 (默认: 0.1)
    - smoothing: 平滑参数 (默认: 0.0)
    - m_steps: 最大化损失的步数 (默认: 1)
    - unlearn_epochs: unlearning轮数 (默认: 30)
    - lr: 学习率 (默认: 0.001)
    """
    # 可调整的参数
    T = 1
    scrub_beta = 0.0
    scrub_gamma = 0.99
    scrub_alpha = 0.1
    smoothing = 0.0
    m_steps = 1
    unlearn_epochs = 50
    lr = 0.001

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
    optimizer = optim.Adam(trainable_list.parameters(), lr=lr, weight_decay=5e-4)

    module_list.append(model_t)

    if torch.cuda.is_available():
        module_list.to(device)
        criterion_list.to(device)
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True

    def avg_fn(averaged_model_parameter, model_parameter, num_averaged):
        return (1 - scrub_beta) * averaged_model_parameter + scrub_beta * model_parameter

    swa_model = torch.optim.swa_utils.AveragedModel(model_s, avg_fn=avg_fn)
    swa_model.to(device)

    for epoch in range(1, unlearn_epochs + 1):
        maximize_loss = 0
        if epoch <= m_steps:
            maximize_loss = train_distill(forget_loader, module_list, swa_model,
                                          criterion_list, optimizer, scrub_gamma, scrub_alpha, scrub_beta, smoothing, "maximize", device,
                                          quiet=False)
        train_acc, train_loss = train_distill(retain_loader, module_list, swa_model, criterion_list,
                                              optimizer, scrub_gamma, scrub_alpha, scrub_beta, smoothing,
                                              "minimize", device, quiet=False)

    return model_s


def adjust_learning_rate(epoch, learning_rate, lr_decay_epochs, lr_decay_rate, optimizer):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    steps = np.sum(epoch > np.asarray(lr_decay_epochs))
    new_lr = learning_rate
    if steps > 0:
        new_lr = learning_rate * (lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    return new_lr, optimizer


# ==================== GA (Gradient Ascent) Unlearning Methods ====================

def gradient_ascent(unlearned_model, forget_loader, optimizer, criterion, args):
    """梯度上升，用于最大化forget set的损失"""
    for batch in forget_loader:
        if isinstance(batch, dict):
            data = {k: v.to(args['device']) for k, v in batch.items() if k != 'labels'}
            target = batch['labels'].to(args['device'])
        else:
            data, target = batch
            data, target = data.to(args['device']), target.to(args['device'])
        
        optimizer.zero_grad()
        output = unlearned_model.forward_propagation(data)
        loss = -criterion(output, target)  # 反向损失
        loss.backward()
        optimizer.step()


def refine_remained(unlearned_model, retain_loader, optimizer, criterion, args):
    """精炼保留数据，用于最小化retain set的损失"""
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


def GA_train(unlearned_model, forget_loader, retain_loader, test_loader, args):
    """
    使用Gradient Ascent方法进行unlearning训练
    可以调整的参数：
    - unlearn_epoch: unlearning轮数 (默认: 30，根据数据集自动调整)
    - lr_ascent: 梯度上升学习率 (默认: args['lr']*0.1，根据数据集自动调整)
    - lr_remained: 保留数据学习率 (默认: args['lr'])
    - weight_decay: 权重衰减 (默认: 5e-4)
    """
    # 可调整的参数
    unlearn_epoch = 40
    optimizer_ascent = optim.Adam(unlearned_model.parameters(), lr=args['lr']*0.04, weight_decay=5e-4)
    optimizer_remained = optim.Adam(unlearned_model.parameters(), lr=args['lr'], weight_decay=5e-4)
    
    # 根据数据集自动调整参数
    if args['dataset_name'] == 'cinic10' or args['dataset_name'] == 'tinyimagenet':
        optimizer_ascent = optim.Adam(unlearned_model.parameters(), lr=args['lr'] * 0.05, weight_decay=5e-4)
    if args['dataset_name'] == 'sst5':
        optimizer_ascent = optim.Adam(unlearned_model.parameters(), lr=args['lr']* 2.0, weight_decay=5e-4)
        unlearn_epoch = 10

    if args['dataset_name'] == 'news20':
        optimizer_ascent = optim.Adam(unlearned_model.parameters(), lr=args['lr']* 2.0, weight_decay=5e-4)
        unlearn_epoch = 12

    if args['dataset_name'] == 'rte':
        optimizer_ascent = optim.Adam(unlearned_model.parameters(), lr=args['lr']* 2.0, weight_decay=5e-4)
        unlearn_epoch = 5
    if args['dataset_name'] == 'mrpc':
        optimizer_ascent = optim.Adam(unlearned_model.parameters(), lr=args['lr']* 2.0, weight_decay=5e-4)
        unlearn_epoch = 10
    
    criterion = nn.CrossEntropyLoss()
    unlearned_model.train()

    for t in range(unlearn_epoch):
        gradient_ascent(unlearned_model, forget_loader, optimizer_ascent, criterion, args)
        retain_set_acc = unlearned_model.test_model_acc(retain_loader)
        forget_set_acc = unlearned_model.test_model_acc(forget_loader)
        test_acc = unlearned_model.test_model_acc(test_loader)
        print('epoch %s: forget set acc (UA) %s  | retain set acc (RA) %s |test acc (TA) %s ' % (t,  round(forget_set_acc, 4), round(retain_set_acc, 4),  round(test_acc, 4)))

        refine_remained(unlearned_model, retain_loader, optimizer_remained, criterion, args)
        retain_set_acc = unlearned_model.test_model_acc(retain_loader)
        forget_set_acc = unlearned_model.test_model_acc(forget_loader)
        test_acc = unlearned_model.test_model_acc(test_loader)
        print('epoch %s: forget set acc (UA) %s  | retain set acc (RA) %s |test acc (TA) %s ' % (t,  round(forget_set_acc, 4), round(retain_set_acc, 4),  round(test_acc, 4)))

    return unlearned_model

