import os
import random

import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from attack.Double_Attack import attack_feature_base, MLP2Layer
from attack.utils import baseline_prep_for_double_attack
from data.load_data import get_data
from data.prepare_data import construct_dataset, split_dataset, split_dataset2, split_dataset3
from model.DNN import DNN
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data import ConcatDataset
from torchvision import datasets, transforms
from scipy.stats import pearsonr

from parameter_parser import parameter_parser
from unlearning.utils import sample_target_samples, save_output, calculate_confidence_with_subsets, TransformedDataset, \
    sample_target_samples2, get_top_bottom_n_indices, add_gaussian_noise


def retrain(args):
     retrain_save_target_for_population_attack_batch(args)
     retrain_save_shadow_for_population_attack_batch(args)

# just save posterior
def retrain_save_target_for_population_attack_batch(args):
    print("dataset and net_name:",args['dataset_name'],args['net_name'])

    train_data, test_data = get_data(args['dataset_name'], model_name=args.get('net_name'))

    target_m,shadow_m,shadow_um = split_dataset(train_data, args['random'])

    train_loader = torch.utils.data.DataLoader(
        target_m, batch_size=args['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args['batch_size'], shuffle=True)

    original_model = DNN(args)
    original_model.train_model(train_loader, test_loader)
    acc_train_loader = original_model.test_model_acc(train_loader)
    acc_test_loader = original_model.test_model_acc(test_loader)
    print(acc_train_loader,acc_test_loader)

    for t in range(args['trials']):
        print(f'The {t}-th trails')

        # unlearned model
        forget_set, retain_set = sample_target_samples(target_m, args['proportion_of_group_unlearn'],args['dataset_name'],False)
        retain_loader = torch.utils.data.DataLoader(
            retain_set, batch_size=args['batch_size'], shuffle=True)

        unlearned_model = DNN(args)
        unlearned_model.train_model(retain_loader, test_loader)
        save_output('target', args, original_model, unlearned_model, forget_set, retain_set, test_data,shadow_um,t)


def retrain_save_shadow_for_population_attack_batch(args):
    print("dataset and net_name:",args['dataset_name'],args['net_name'])
    train_data, test_data = get_data(args['dataset_name'], model_name=args.get('net_name'))

    target_m,shadow_m, shadow_um = split_dataset(train_data, args['random'])

    train_loader = torch.utils.data.DataLoader(
        shadow_m, batch_size=args['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args['batch_size'], shuffle=True)

    original_model = DNN(args)
    original_model.train_model(train_loader, test_loader)
    for t in range(args['observations']):
        print(f'The {t}-th observations')

        # unlearned model
        forget_set, retain_set = sample_target_samples(shadow_m, args['proportion_of_group_unlearn'],args['dataset_name'],False)
        retain_loader = torch.utils.data.DataLoader(
            retain_set, batch_size=args['batch_size'], shuffle=True)

        unlearned_model = DNN(args)
        unlearned_model.train_model(retain_loader, test_loader)
        save_output('shadow', args, original_model, unlearned_model, forget_set, retain_set, test_data,shadow_um,t)

# for well generalized to overfitting
def retrain_save_target_for_population_attack_batch2(args):
    for t in range(args['trials']):
        print(f'The {t}-th trial')
        A=['well-well','well-over','over-well','over-over']
        A=['well-well']
        train_data, test_data = get_data(args['dataset_name'], model_name=args.get('net_name'))
      #  target_m, shadow_m, shadow_um = split_dataset(train_data, args['random'])

        target_m, shadow_um = train_test_split(train_data, test_size=0.5, random_state=args['random'])

        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args['batch_size'], shuffle=False)

        target_m_transformed = TransformedDataset(target_m)

        for a in A:
            print(f'The type is {a} -------------------')

            # 确定 train_loader 的数据来源
            if a in ['well-over', 'well-well']:
                train_loader = torch.utils.data.DataLoader(target_m_transformed, batch_size=args['batch_size'], shuffle=True)

            else:
                train_loader = torch.utils.data.DataLoader(target_m, batch_size=args['batch_size'], shuffle=True)

            # 训练原始模型
            original_model = DNN(args)
            original_model.train_model(train_loader, test_loader)


            # 构造 remaining_loader
            if a == 'well-well' :


                target_sample, remaining_data = sample_target_samples(target_m_transformed, args['proportion_of_group_unlearn'],
                                                                      args['dataset_name'], False)
                remaining_loader = torch.utils.data.DataLoader(remaining_data, batch_size=args['batch_size'],
                                                               shuffle=True)

            elif a=='over-well':
                target_sample2, remaining_data2,target_sample,remaining_data = sample_target_samples2(target_m_transformed,target_m, args['proportion_of_group_unlearn'],
                                                                      args['dataset_name'], False)

                remaining_loader = torch.utils.data.DataLoader(remaining_data2, batch_size=args['batch_size'],
                                                               shuffle=True)

            else:
                target_sample, remaining_data = sample_target_samples(target_m, args['proportion_of_group_unlearn'],
                                                                      args['dataset_name'], False)
                remaining_loader = torch.utils.data.DataLoader(remaining_data, batch_size=args['batch_size'],
                                                               shuffle=True)

            # 训练 unlearned model
            unlearned_model = DNN(args)
            unlearned_model.train_model(remaining_loader, test_loader)

            # 保存实验结果
            save_output(a, args, original_model, unlearned_model, target_sample, remaining_data, test_data, shadow_um,t)

# for different confident samples
def retrain_save_target_for_population_attack_batch3(args):
    for t in range(args['trials']):
        print(f'The {t}-th trails')
        train_data, test_data = get_data(args['dataset_name'], model_name=args.get('net_name'))

       # target_m, target_um, shadow_m, shadow_um = split_dataset(train_data, args['random'])
        target_m,  target_um,shadow_m, shadow_um = split_dataset2(train_data, args['random'])

        train_loader = torch.utils.data.DataLoader(
            target_m, batch_size=args['batch_size'], shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            target_um, batch_size=args['batch_size'], shuffle=True)
        original_model = DNN(args)
        original_model.train_model(train_loader, test_loader)
        K=int(len(target_m)*args['proportion_of_group_unlearn'])
        high_confidence_subset, remaining_after_high,low_confidence_subset, remaining_after_low,high_entropy_subset,remaining_after_high_entropy,low_entropy_subset, remaining_after_low_entropy,random_subset,remaining_after_random=calculate_confidence_with_subsets(original_model, train_loader,K, args['device'])

        for flag in ['random','low_conf','high_conf','low_entropy','high_entropy']:
            print(flag)
            if flag == 'low_conf':
                target_sample, remaining_data = low_confidence_subset, remaining_after_low
            elif flag == 'high_conf':
                target_sample, remaining_data = high_confidence_subset, remaining_after_high
            elif flag =='low_entropy':
                target_sample, remaining_data = low_entropy_subset, remaining_after_low_entropy
            elif flag =='high_entropy':
                target_sample, remaining_data = high_entropy_subset, remaining_after_high_entropy
            else :
                target_sample, remaining_data = random_subset,remaining_after_random

            remaining_loader = torch.utils.data.DataLoader(
                remaining_data, batch_size=args['batch_size'], shuffle=True)

            unlearned_model = DNN(args)
            unlearned_model.train_model(remaining_loader, test_loader)
            save_output(flag, args, original_model, unlearned_model, target_sample, remaining_data, target_um,shadow_um,t)

#ASR
def retrain_save_target_for_population_attack_batch4(args):
    train_data, test_data = get_data(args['dataset_name'], model_name=args.get('net_name'))
    target_m, target_um, shadow_m, shadow_um = split_dataset2(train_data, args['random'])

    P_shadow_original, P_shadow_unlearned, P_original_target, P_unlearned_target, mem_train, mem_test, train_sample_label, test_sample_label = baseline_prep_for_double_attack(
        args)
    # drop the second dimension
    P_shadow_original = P_shadow_original[:, 0, :]

    # first attack model
    attack_X = []
    for posterior_shadow, label in zip(P_shadow_original, train_sample_label):
        attack_X.append([posterior_shadow[label]])
    attack_X = np.array(attack_X)
    ytest = mem_train
    Atrain, Aval, y_train, y_val = train_test_split(attack_X, ytest, test_size=0.2, random_state=args['random'])


    clf = MLP2Layer(in_dim=attack_X.shape[1], out_dim=2, layer_list=128, device=torch.device(args['device']))
    clf.criterion = torch.nn.CrossEntropyLoss()
    clf.optimizer = torch.optim.Adam(clf.parameters(), lr=0.001, weight_decay=1e-5)
    clf.to(torch.device(args['device']))
    clf.train_epochs(Atrain, y_train, Aval, y_val, epochs=100, max_patient=20)
    ASR_list=[]
    for t in range(20):  #get ASR
        print(f'The {t}-th observations')

        train_loader = torch.utils.data.DataLoader(
            target_m, batch_size=args['batch_size'], shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            target_um, batch_size=args['batch_size'], shuffle=True)
        original_model = DNN(args)
        original_model.train_model(train_loader, test_loader)
        original_model_posterior_list=[]
        #得到输出
        for i in range(len(target_m)):
            data, label = target_m[i]  # 直接从数据集中索引
            original_model_posterior = original_model.predict_proba(data)
            original_model_posterior_list.append([original_model_posterior[0][label]])

        # Testing original model
        probs_original = clf.pred(original_model_posterior_list)
        ASR= probs_original[:, 1].tolist()
        ASR_list.append(ASR)

    avg_ASR = [sum(col) / len(col) for col in zip(*ASR_list)]

    print('average ASR: ', avg_ASR)

    for t in range(args['trials']):
        print(f'The {t}-th trials')
        train_loader = torch.utils.data.DataLoader(
            target_m, batch_size=args['batch_size'], shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            target_um, batch_size=args['batch_size'], shuffle=True)
        original_model = DNN(args)
        original_model.train_model(train_loader, test_loader)

        K=int(len(target_m)*args['proportion_of_group_unlearn'])

        top_n_indices, bottom_n_indices=get_top_bottom_n_indices(avg_ASR,K)

        highest_ASR_samples = Subset(target_m, top_n_indices)  # 取出 top_n_idx 对应的样本
        highest_ASR_remaining_indices = list(set(range(len(target_m))) - set(top_n_indices))  # 计算剩余样本索引
        highest_ASR_remaining_subset = Subset(target_m, highest_ASR_remaining_indices)  # 取出剩余样本

        lowest_ASR_samples = Subset(target_m, bottom_n_indices)
        lowest_ASR_remaining_indices = list(set(range(len(target_m))) - set(bottom_n_indices))  # 计算剩余样本索引
        lowest_ASR_remaining_subset = Subset(target_m, lowest_ASR_remaining_indices)  # 取出剩余样本

        high_confidence_subset, remaining_after_high, low_confidence_subset, remaining_after_low, high_entropy_subset, remaining_after_high_entropy, low_entropy_subset, remaining_after_low_entropy, random_subset, remaining_after_random = calculate_confidence_with_subsets(
            original_model, train_loader, K, args['device'])

       # for flag in ['random', 'low_conf', 'high_conf', 'low_entropy', 'high_entropy','low_asr','high_asr']:
        for flag in ['low_asr', 'high_asr']:

            print(flag)
            if flag == 'low_conf':
                target_sample, remaining_data = low_confidence_subset, remaining_after_low
            elif flag == 'high_conf':
                target_sample, remaining_data = high_confidence_subset, remaining_after_high
            elif flag == 'low_entropy':
                target_sample, remaining_data = low_entropy_subset, remaining_after_low_entropy
            elif flag == 'high_entropy':
                target_sample, remaining_data = high_entropy_subset, remaining_after_high_entropy
            elif flag == 'low_asr':
                target_sample, remaining_data = lowest_ASR_samples, lowest_ASR_remaining_subset
            elif flag == 'high_asr':
                target_sample, remaining_data = highest_ASR_samples, highest_ASR_remaining_subset
            else:
                target_sample, remaining_data = random_subset, remaining_after_random

            remaining_loader = torch.utils.data.DataLoader(
                remaining_data, batch_size=args['batch_size'], shuffle=True)

            unlearned_model = DNN(args)
            unlearned_model.train_model(remaining_loader, test_loader)

            save_output(flag, args, original_model, unlearned_model, target_sample, remaining_data, target_um,shadow_um,t)




def retrain_save_target_for_population_attack_batch6(args):
    train_data, test_data = get_data(args['dataset_name'], model_name=args.get('net_name'))
    target_m, target_um, shadow_m, shadow_um = split_dataset2(train_data, args['random'])

    K = int(0.02 * len(target_m))
    outlier_indices = np.random.choice(len(target_m), K, replace=False)

    inlier_indices_all=list(set(range(len(target_m))) - set(outlier_indices))
    inlier_indices = np.random.choice(inlier_indices_all, size=K, replace=False)

    inlier_indices_remaining = list(set(range(len(target_m))) - set(inlier_indices))

    new_target_m = []

    for idx in range(len(target_m)):
        img, label = target_m[idx]
        if idx in outlier_indices:
            noisy_img = add_gaussian_noise(img, std=5.0)
            label=10
            new_target_m.append((noisy_img, label))  # 保留原始标签
        else:
            new_target_m.append((img.clone(), label))  # 保持原样本

    for t in range(args['trials']):
        print(f'The {t}-th trials')
        train_loader = torch.utils.data.DataLoader(
            new_target_m, batch_size=args['batch_size'], shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            target_um, batch_size=args['batch_size'], shuffle=True)
        original_model = DNN(args)
        original_model.train_model(train_loader, test_loader)

        outlier_indices_samples = Subset(new_target_m, outlier_indices)  # 取出 outlier_indices 对应的样本
        outlier_indices_remaining_subset = Subset(new_target_m, inlier_indices_all)  # 取出剩余样本

        inlier_indices_samples = Subset(new_target_m, inlier_indices)
        inlier_indices_remaining_samples = Subset(new_target_m, inlier_indices_remaining)


        for flag in ['in', 'out']:

            print(flag)
            if flag == 'in':
                target_sample, remaining_data = inlier_indices_samples, inlier_indices_remaining_samples
            elif flag == 'out':
                target_sample, remaining_data = outlier_indices_samples, outlier_indices_remaining_subset

            remaining_loader = torch.utils.data.DataLoader(
                remaining_data, batch_size=args['batch_size'], shuffle=True)

            unlearned_model = DNN(args)
            unlearned_model.train_model(remaining_loader, test_loader)

            save_output(flag, args, original_model, unlearned_model, target_sample, remaining_data, target_um,shadow_um,t)





# for pearson correlation between uncerainty and margin
def retrain_save_target_for_population_attack_batch5(args):
    train_data, test_data = get_data(args['dataset_name'], model_name=args.get('net_name'))
    target_m, target_um, shadow_m, shadow_um = split_dataset2(train_data, args['random'])

    P_shadow_original, P_shadow_unlearned, P_original_target, P_unlearned_target, mem_train, mem_test, train_sample_label, test_sample_label = baseline_prep_for_double_attack(args)
    # drop the second dimension
    P_shadow_original = P_shadow_original[:, 0, :]

    # first attack model
    attack_X = []
    for posterior_shadow, label in zip(P_shadow_original, train_sample_label):
        attack_X.append([posterior_shadow[label]])
    attack_X = np.array(attack_X)
    ytest = mem_train
    Atrain, Aval, y_train, y_val = train_test_split(attack_X, ytest, test_size=0.2, random_state=args['random'])


    clf = MLP2Layer(in_dim=attack_X.shape[1], out_dim=2, layer_list=128, device=torch.device(args['device']))
    clf.criterion = torch.nn.CrossEntropyLoss()
    clf.optimizer = torch.optim.Adam(clf.parameters(), lr=0.001, weight_decay=1e-5)
    clf.to(torch.device(args['device']))
    clf.train_epochs(Atrain, y_train, Aval, y_val, epochs=100, max_patient=20)
    confidences = []  # 存储 Top-2 置信度差值
    entropies = []  # 存储信息熵
    for t in range(args['trials']):
        print(f'The {t}-th trials')
        train_loader = torch.utils.data.DataLoader(
            target_m, batch_size=args['batch_size'], shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            target_um, batch_size=args['batch_size'], shuffle=True)
        original_model = DNN(args)
        original_model.train_model(train_loader, test_loader)

        indices = list(range(len(train_loader.dataset)))

        inputs = torch.stack([train_loader.dataset[i][0] for i in indices]).to(args['device'])
        outputs = original_model(inputs)

        probabilities = torch.softmax(outputs, dim=1)

        # 计算 Top-2 confidence
        top2_probs, _ = torch.topk(probabilities, k=2, dim=1)
        confidence = top2_probs[:, 0] - top2_probs[:, 1]
        confidences.extend(confidence.cpu().detach().numpy())

        # 计算信息熵
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-10), dim=1)  # 避免 log(0)
        entropies.extend(entropy.cpu().detach().numpy())


    sorted_confidence_values = sorted(confidences)
    sorted_entropies_values = sorted(entropies)
    sorted_confidence_values=np.array(sorted_confidence_values)
    sorted_entropies_values=np.array(sorted_entropies_values)

    # 计算皮尔逊相关系数和p值
    r, p_value = pearsonr(sorted_confidence_values, sorted_entropies_values)

    print(f"相关系数 r = {r:.3f}, p值 = {p_value:.4f}")

    exit()



def retrain_save_target_for_population_attack_batch7(args):
    print("dataset and net_name:",args['dataset_name'],args['net_name'])

    train_data, test_data = get_data(args['dataset_name'], model_name=args.get('net_name'))

    target_m,shadow_m,shadow_um = split_dataset(train_data, args['random'])

    train_loader = torch.utils.data.DataLoader(
        target_m, batch_size=args['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args['batch_size'], shuffle=True)

    original_model = DNN(args)
    original_model.train_model(train_loader, test_loader)
    acc_train_loader = original_model.test_model_acc(train_loader)
    acc_test_loader = original_model.test_model_acc(test_loader)
    print(acc_train_loader,acc_test_loader)
    for t in range(args['trials']):
        print(f'The {t}-th trails')

        # unlearned model
        forget_set, retain_set = sample_target_samples(target_m, args['proportion_of_group_unlearn'],args['dataset_name'],False)
        retain_loader = torch.utils.data.DataLoader(
            retain_set, batch_size=args['batch_size'], shuffle=True)

        unlearned_model = DNN(args)
        unlearned_model.train_model(retain_loader, test_loader)
        save_output('target', args, original_model, unlearned_model, forget_set, retain_set, test_data,shadow_um,t)


# size of shadow set
def retrain_save_shadow_for_population_attack2(args):
    train_data, test_data = get_data(args['dataset_name'], model_name=args.get('net_name'))

    target_m, shadow_m, shadow_um = split_dataset(train_data, args['random'])
    P=args['size_of_shadow_training']
    if args['size_of_shadow_training']!=-1:
        shadow_m, _ = train_test_split(shadow_m, test_size=1-P)
        shadow_um, _ = train_test_split(shadow_um, test_size=1-P)
    flag=f'shadow_{P}'

    train_loader = torch.utils.data.DataLoader(
        shadow_m, batch_size=args['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args['batch_size'], shuffle=False)

    original_model = DNN(args)
    original_model.train_model(train_loader, test_loader)
    for t in range(args['observations']):
        print(f'The {t}-th observations')
        # unlearned model
        target_sample, remaining_data = sample_target_samples(shadow_m,args['proportion_of_group_unlearn'], args['dataset_name'],False)

        remaining_loader = torch.utils.data.DataLoader(
            remaining_data, batch_size=args['batch_size'], shuffle=True)
        unlearned_model = DNN(args)
        unlearned_model.train_model(remaining_loader, test_loader)

        save_output(flag, args, original_model, unlearned_model, target_sample, remaining_data, test_data,
                    shadow_um, t)


