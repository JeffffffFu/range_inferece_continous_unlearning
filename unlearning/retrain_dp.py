import os
import random

import pandas as pd
from sklearn.model_selection import train_test_split

from data.load_data import get_data
from data.prepare_data import construct_dataset, split_dataset
from model.DNN import DNN
import torch
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager

from model.ResNet import resnet18, resnet18_dp
from parameter_parser import parameter_parser
from unlearning.utils import sample_target_samples, save_output
from utils.compute_dp_sgd import apply_dp_sgd_analysis
from utils.dp_optimizer import get_dp_optimizer
from utils.sampling import get_data_loaders_possion
import torch.optim as optim
from torch.utils.data import TensorDataset
from opacus.validators import ModuleValidator
from opacus.utils.uniform_sampler import UniformWithReplacementSampler

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# 获取当前时间并格式化为唯一字符串
def retrain_dp(args):
    retrain_dp_save_target_for_population_attack(args)
   # retrain_dp_save_shadow_for_population_attack(args)


#opacus
# def retrain_dp_save_target_for_population_attack(args): #opacus
#
#     train_data, test_data = get_data(args['dataset_name'], augment=False)
#
#     target_m, target_um, shadow_m, shadow_um = split_dataset(train_data, args['random'])
#     minibatch_loader, microbatch_loader = get_data_loaders_possion(minibatch_size=args['batch_size'],microbatch_size=1,iterations=1)
#
#     test_loader = torch.utils.data.DataLoader(
#         test_data, batch_size=args['batch_size'], shuffle=False,drop_last=True)
#     train_loader = torch.utils.data.DataLoader(
#         train_data, batch_size=args['batch_size'], shuffle=False)
#     orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 64))+ [128, 256, 512]
#     C=1.0
#     delta=1e-5
#     sigma=args['sigma']
#
#
#     original_model=DNN(args)
#     original_model=original_model.to(args['device'])
#     optimizer = optim.Adam(original_model.parameters(), lr=args['lr'], weight_decay=1e-4)
#
#
#
#     privacy_engine = PrivacyEngine()
#     original_model, optimizer, train_loader = privacy_engine.make_private(
#         module=original_model,
#         optimizer=optimizer,
#         data_loader=train_loader,
#         noise_multiplier=0.5,
#         max_grad_norm=1.0,
#     )
#
#     for epoch in range(args['num_epochs']):
#
#         train(epoch,original_model, train_loader, test_loader,optimizer, privacy_engine,args)
#
#     #unlearned model
#     for t in range(args['trials']):
#         print(f'The {t}-th trails')
#
#         #unlearned model
#         target_sample, remaining_data = sample_target_samples(target_m, args['proportion_of_group_unlearn'], args['dataset_name'],False)
#
#         unlearned_model = DNN(args)
#         unlearned_model = unlearned_model.to(args['device'])
#         optimizer = get_dp_optimizer(lr=args['lr'], C_t=C, sigma=sigma, batch_size=args['batch_size'], model=unlearned_model)
#         for iter in range(args['num_epochs']):
#             epsilon, best_alpha = apply_dp_sgd_analysis(args['batch_size'] / len(train_data), sigma, iter, orders,
#                                                         delta)  # comupte privacy cost
#             train_dl = minibatch_loader(target_m)  # possion sampling
#             for id, (data, target) in enumerate(train_dl):
#                 optimizer.minibatch_size = len(data)
#
#             train_with_dp(unlearned_model, train_dl, optimizer, args['device'])
#             test_loss, test_accuracy = validation(original_model, test_loader, args['device'])
#             train_loss, train_acc = validation(original_model, train_dl, args['device'])
#             print(
#                 f'iters:{iter},'f'epsilon:{epsilon:.4f} |'f' train_acc: ({train_acc:.2f}%),'f' test_accuracy:({test_accuracy:.2f}%)')
#
#         save_output('target', args, original_model, unlearned_model, target_sample, remaining_data, test_data,
#                     shadow_um, t)
#
def retrain_dp_save_target_for_population_attack(args):

    train_data, test_data = get_data(args['dataset_name'], model_name=args.get('net_name'))

    target_m, shadow_m, shadow_um = split_dataset(train_data, args['random'])
    minibatch_loader, microbatch_loader = get_data_loaders_possion(minibatch_size=args['batch_size'],microbatch_size=1,iterations=1)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args['batch_size'], shuffle=False)
    train_loader = torch.utils.data.DataLoader(
        target_m, batch_size=args['batch_size'], shuffle=False)
    orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 64))+ [128, 256, 512]
    C=args['C']
    delta=5*1e-4
    sigma=args['sigma']

    original_model = DNN(args)
    original_model=original_model.to(args['device'])
    optimizer = get_dp_optimizer(lr=args['lr'], C_t=C, sigma=sigma, batch_size=args['batch_size'], model=original_model)
    optimizer.minibatch_size = args['batch_size']
    flag=f'dp_{sigma}'

    for epoch in range(args['num_epochs']):
        epsilon, best_alpha = apply_dp_sgd_analysis(args['batch_size'] / len(target_m), sigma, (epoch+1)*len(train_loader), orders, delta) #comupte privacy cost
     #   train_dl = minibatch_loader(target_m)  # possion sampling
     #    for id, (data, target) in enumerate(train_dl):
     #        optimizer.minibatch_size = len(data)

        train_with_dp(original_model, train_loader, optimizer, args['device'])
        test_loss, test_accuracy =validation(original_model, test_loader,args['device'])
        train_loss, train_acc =validation(original_model, train_loader,args['device'])

        print(
            f'epoches:{epoch},'f'epsilon:{epsilon:.4f} |'f' train_acc: ({train_acc:.2f}%),'f' test_accuracy:({test_accuracy:.2f}%)')

    #unlearned model
    for t in range(args['trials']):
        print(f'The {t}-th trails')

        #unlearned model
        target_sample, remaining_data = sample_target_samples(target_m, args['proportion_of_group_unlearn'], args['dataset_name'],False)

        unlearned_model = DNN(args)
        unlearned_model = unlearned_model.to(args['device'])
        optimizer = get_dp_optimizer(lr=args['lr'], C_t=C, sigma=sigma, batch_size=args['batch_size'], model=unlearned_model)
        remaining_loader = torch.utils.data.DataLoader(
            remaining_data, batch_size=args['batch_size'], shuffle=False)
        for epoch in range(args['num_epochs']):
            epsilon, best_alpha = apply_dp_sgd_analysis(args['batch_size'] / len(target_m), sigma, (epoch+1)*len(train_loader), orders,
                                                        delta)  # comupte privacy cost

            train_with_dp(unlearned_model, remaining_loader, optimizer, args['device'])
            test_loss, test_accuracy = validation(unlearned_model, test_loader, args['device'])
            train_loss, train_acc = validation(unlearned_model, remaining_loader, args['device'])
            print(
                f'epoch:{epoch},'f'epsilon:{epsilon:.4f} |'f' train_acc: ({train_acc:.2f}%),'f' test_accuracy:({test_accuracy:.2f}%)')

        save_path = os.getcwd() + f"/save/{args['U_method']}/{args['net_name']}/{args['dataset_name']}/{args['proportion_of_group_unlearn']}/{args['sigma']}/target/{t}/"
        os.makedirs(save_path, exist_ok=True)

        save_output(flag, args, original_model, unlearned_model, target_sample, remaining_data, test_data,
                    shadow_um, t)


def train_with_dp(model, train_loader, optimizer, device):
    model=model.to(device)
    correct = 0
    for id, batch in enumerate(train_loader):
        # 处理不同类型的数据
        if isinstance(batch, dict):
            # 文本数据（SST5等）
            data = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            target = batch['labels'].to(device)
        else:
            # 图像数据
            data, target = batch
            data, target = data.to(device), target.to(device)
        
        optimizer.zero_accum_grad()
        
        # 为文本数据和图像数据创建不同的微批次处理
        if isinstance(data, dict):
            # 文本数据的微批次处理
            batch_size = target.size(0)
            for i in range(batch_size):
                optimizer.zero_microbatch_grad()
                
                # 创建单样本输入
                single_data = {k: v[i:i+1] for k, v in data.items()}
                single_target = target[i:i+1]
                
                output = model(single_data)
                loss = F.cross_entropy(output, single_target)
                
                loss.backward()
                optimizer.microbatch_step()
        else:
            # 图像数据的微批次处理（原有逻辑）
            for iid, (X_microbatch, y_microbatch) in enumerate(TensorDataset(data, target)):
                optimizer.zero_microbatch_grad()
                output = model(torch.unsqueeze(X_microbatch, 0))

                if len(output.shape) == 2:
                    output = torch.squeeze(output, 0)
                loss = F.cross_entropy(output, y_microbatch)

                loss.backward()
                optimizer.microbatch_step()
        
        optimizer.step_dp()

def validation(model, test_loader, device):
    model.eval()
    num_examples = 0
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch in test_loader:
            # 处理不同类型的数据
            if isinstance(batch, dict):
                # 文本数据（SST5等）
                data = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
                target = batch['labels'].to(device)
                num_examples += target.size(0)
            else:
                # 图像数据
                data, target = batch
                data, target = data.to(device), target.to(device)
                num_examples += len(data)
            
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum')

            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
            
    test_loss /= num_examples
    test_acc = 100. * correct / num_examples

    return test_loss, test_acc