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

def GA(args):
     GA_save_target_for_population_attack(args)
     GA_save_shadow_for_population_attack(args)


def GA_save_target_for_population_attack(args):
    train_data, test_data = get_data(args['dataset_name'], model_name=args.get('net_name'))
    print("dataset and net_name:----------",args['dataset_name'],args['net_name'])

    target_m,  shadow_m, shadow_um = split_dataset(train_data, args['random'])

    # 为文本数据创建自定义collate函数
    def collate_fn(batch):
        """自定义collate函数，处理文本和图像数据"""
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
   # original_model.load_state_dict(torch.load(f"{args['dataset_name']}_original_model.pth", map_location=args['device']))
   # original_model.to(args['device'])
    original_model.train_model(train_loader, test_loader)


    for t in range(args['trials']):
        print(f'The {t}-th trails')
        forget_set, retain_set = sample_target_samples(target_m, args['proportion_of_group_unlearn'], args['dataset_name'],False)

        def collate_fn(batch):
            """自定义collate函数，处理文本和图像数据"""
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
            forget_set, batch_size=len(forget_set), shuffle=False, collate_fn=collate_fn)
        retain_loader = torch.utils.data.DataLoader(
            retain_set, batch_size=args['batch_size'], shuffle=False, collate_fn=collate_fn)
        unlearned_model = DNN(args)
        unlearned_model.load_state_dict(original_model.state_dict())

        unlearned_model=GA_train(unlearned_model, forget_loader,retain_loader, test_loader, args)

        save_output('target', args, original_model, unlearned_model, forget_set, retain_set, test_data,shadow_um,t)



def GA_save_shadow_for_population_attack(args):
    print("dataset and net_name:----------",args['dataset_name'],args['net_name'])

    train_data, test_data = get_data(args['dataset_name'], model_name=args.get('net_name'))

    target_m,shadow_m, shadow_um = split_dataset(train_data, args['random'])
    # 为文本数据创建自定义collate函数
    def collate_fn(batch):
        """自定义collate函数，处理文本和图像数据"""
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
        def collate_fn(batch):
            """自定义collate函数，处理文本和图像数据"""
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
        unlearned_model=GA_train(unlearned_model, forget_loader,retain_loader, test_loader, args)

        save_output('shadow', args, original_model, unlearned_model, forget_set, retain_set, test_data,shadow_um,t)



def gradient_ascent(unlearned_model,forget_loader,optimizer,criterion,args):

        for batch in forget_loader:
            if isinstance(batch, dict):
                data = {k: v.to(args['device']) for k, v in batch.items() if k != 'labels'}
                target = batch['labels'].to(args['device'])
            else:
                data, target = batch
                data, target = data.to(args['device']), target.to(args['device'])
            
            optimizer.zero_grad()
            output = unlearned_model.forward_propagation(data)
            loss = -criterion(output, target)  # iverse loss
            loss.backward()
            optimizer.step()

def refine_remained(unlearned_model,retain_loader,optimizer,criterion,args):

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

def GA_train(unlearned_model,forget_loader,retain_loader,test_loader,args):
    unlearn_epoch=30
    optimizer_ascent = optim.Adam(unlearned_model.parameters(), lr=args['lr']*0.1, weight_decay=5e-4)
    optimizer_remained = optim.Adam(unlearned_model.parameters(), lr=args['lr'], weight_decay=5e-4)
    if args['dataset_name']=='cinic10' or args['dataset_name']=='tinyimagenet':
        optimizer_ascent = optim.Adam(unlearned_model.parameters(), lr=args['lr'] * 0.05, weight_decay=5e-4)
    if args['dataset_name'] == 'sst5':
        optimizer_ascent = optim.Adam(unlearned_model.parameters(), lr=args['lr']* 2.0, weight_decay=5e-4)
        unlearn_epoch = 10

    if  args['dataset_name'] == 'news20':
        optimizer_ascent = optim.Adam(unlearned_model.parameters(), lr=args['lr']* 2.0, weight_decay=5e-4)
        unlearn_epoch=12

    if args['dataset_name'] == 'rte':
        optimizer_ascent = optim.Adam(unlearned_model.parameters(), lr=args['lr']* 2.0, weight_decay=5e-4)
        unlearn_epoch = 5
    if args['dataset_name'] == 'mrpc':
        optimizer_ascent = optim.Adam(unlearned_model.parameters(), lr=args['lr']* 2.0, weight_decay=5e-4)
        unlearn_epoch = 10
    criterion = nn.CrossEntropyLoss()

    unlearned_model.train()

    for t in range(unlearn_epoch):
        gradient_ascent( unlearned_model, forget_loader, optimizer_ascent, criterion, args)
        retain_set_acc =unlearned_model.test_model_acc(retain_loader)
        forget_set_acc =unlearned_model.test_model_acc(forget_loader)
        test_acc = unlearned_model.test_model_acc(test_loader)
        print('epoch %s: forget set acc (UA) %s  | retain set acc (RA) %s |test acc (TA) %s ' % (t,  round(forget_set_acc, 4), round(retain_set_acc, 4),  round(test_acc, 4)))

        refine_remained( unlearned_model, retain_loader, optimizer_remained, criterion, args)
        retain_set_acc =unlearned_model.test_model_acc(retain_loader)
        forget_set_acc =unlearned_model.test_model_acc(forget_loader)
        test_acc = unlearned_model.test_model_acc(test_loader)
        print('epoch %s: forget set acc (UA) %s  | retain set acc (RA) %s |test acc (TA) %s ' % (t,  round(forget_set_acc, 4), round(retain_set_acc, 4),  round(test_acc, 4)))


    return unlearned_model