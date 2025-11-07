from sklearn.model_selection import train_test_split

from data.load_data import get_data
from data.prepare_data import split_dataset
import torch
import torch.optim as optim
import numpy as np
import torch.nn as nn
from model.DNN import DNN
from unlearning.utils import sample_target_samples, save_output
import os
from itertools import cycle

def NPO(args):
     NPO_save_target_for_population_attack(args)
     NPO_save_shadow_for_population_attack(args)


def NPO_save_target_for_population_attack(args):
    train_data, test_data = get_data(args['dataset_name'], model_name=args.get('net_name'))
    print("dataset and net_name:----------",args['dataset_name'],args['net_name'])

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

    original_model.train_model(train_loader, test_loader)


    for t in range(args['trials']):
        print(f'The {t}-th trails')
        forget_set, retain_set = sample_target_samples(target_m, args['proportion_of_group_unlearn'], args['dataset_name'],False)

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
                # 图像数据，使用默认collate
                return torch.utils.data.dataloader.default_collate(batch)
        
        forget_loader = torch.utils.data.DataLoader(
            forget_set, batch_size=args['batch_size'], shuffle=False, collate_fn=collate_fn)
        retain_loader = torch.utils.data.DataLoader(
            retain_set, batch_size=args['batch_size'], shuffle=False, collate_fn=collate_fn)
        unlearned_model = DNN(args)
        unlearned_model.load_state_dict(original_model.state_dict())

        unlearned_model=NPO_train(original_model, unlearned_model, forget_loader,retain_loader, test_loader, args)
        save_output('target', args, original_model, unlearned_model, forget_set, retain_set, test_data,shadow_um,t)



def NPO_save_shadow_for_population_attack(args):
    train_data, test_data = get_data(args['dataset_name'], model_name=args.get('net_name'))
    target_m, shadow_m, shadow_um = split_dataset(train_data, args['random'])
    print("dataset and net_name:----------",args['dataset_name'],args['net_name'])

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
        shadow_m, batch_size=args['batch_size'], shuffle=True, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args['batch_size'], shuffle=False, collate_fn=collate_fn)

    original_model = DNN(args)
    original_model.train_model(train_loader, test_loader)

    for t in range(args['observations']):
        print(f'The {t}-th observations')

        # unlearned model
        forget_set, retain_set = sample_target_samples(shadow_m, args['proportion_of_group_unlearn'], args['dataset_name'])

        # 为文本数据创建自定义collate函数
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
        
        forget_loader = torch.utils.data.DataLoader(
            forget_set, batch_size=args['batch_size'], shuffle=False, collate_fn=collate_fn)
        retain_loader = torch.utils.data.DataLoader(
            retain_set, batch_size=args['batch_size'], shuffle=False, collate_fn=collate_fn)
        unlearned_model = DNN(args)
        unlearned_model.load_state_dict(original_model.state_dict())
        unlearned_model=NPO_train(original_model, unlearned_model, forget_loader,retain_loader, test_loader, args)

        save_output('shadow', args, original_model, unlearned_model, forget_set, retain_set, test_data,shadow_um,t)



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