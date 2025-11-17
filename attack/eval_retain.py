import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from PIL.ImImagePlugin import number
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import random
from sklearn.metrics import classification_report

from attack.metric.metric import all_metrics_for_three_class, save_metric
from attack.utils import baseline_prep, baseline_prep_for_double_attack, baseline_prep_for_double_attack2
from parameter_parser import parameter_parser
from data.load_data import get_data
from data.prepare_data import construct_dataset, split_dataset
import os
from model.DNN import DNN
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import f1_score, accuracy_score


class MLP2Layer(nn.Module):
    def __init__(self, in_dim, out_dim, layer_list, device):
        super(MLP2Layer, self).__init__()
        torch.manual_seed(3407)

        self.fc1 = nn.Linear(in_dim, layer_list)
        self.fc2 = nn.Linear(layer_list, out_dim)

        self.outdim = out_dim
        self.indim = in_dim

        self.criterion = None
        self.optimizer = None
        self.device = device

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x

    def train_one_epoch(self, Xtrain, ytrain):
        self.train()
        self.optimizer.zero_grad()

        outputs = self(torch.Tensor(Xtrain).to(self.device))
        loss = self.criterion(outputs, torch.LongTensor(ytrain).to(self.device))
        loss.backward()
        self.optimizer.step()

    def train_epochs(self, train_x, train_y, val_x, val_y, epochs, max_patient):
        pbar = tqdm(range(epochs), leave=True, desc=f"Attack Training")
        if val_x is None:
            val_x = train_x
            val_y = train_y
        opt_loss = 1e10
        patient = max_patient
        for i in pbar:
            self.train_one_epoch(train_x, train_y)
            train_loss, train_acc = self.loss_acc(train_x, train_y)
            val_loss, val_acc = self.loss_acc(val_x, val_y)

            pbar.set_postfix({'Loss': train_loss,
                              'Acc': train_acc,
                              'Val Loss': val_loss,
                              'Val Acc': val_acc})
            if opt_loss / 1.001 > val_loss:
                opt_loss = val_loss
                patient = max_patient
            else:
                patient = patient - 1

            if patient == 0:
                pbar.close()
                break

    def loss_acc(self, Xtest, ytest):
        self.eval()
        outputs = self(torch.Tensor(Xtest).to(self.device))
        loss = self.criterion(outputs, torch.LongTensor(ytest).to(self.device))
        acc = (outputs.argmax(dim=1) == torch.LongTensor(ytest).to(self.device)).sum() / len(outputs)
        return loss.cpu().detach().item(), acc.cpu().detach().item()

    def acc(self, X_target, y_target):
        outputs_target = self(torch.Tensor(X_target).to(self.device)).cpu()
        acc = metrics.accuracy_score(y_target, outputs_target.detach().numpy().argmax(axis=1))
        return acc

    def pred(self, X_target):
        '''Predict posteriors'''
        outputs_target = self(torch.Tensor(X_target).to(self.device)).cpu()
        return outputs_target.detach().numpy()


def attack_feature_base(P_shadow, label_list):
    attack_feature = []
    # for posterior_shadow, label in zip(P_shadow, label_list):
    #     attack_feature.append([posterior_shadow[label]])
    for posterior_shadow, label in zip(P_shadow, label_list):
        top_3 = sorted(posterior_shadow, reverse=True)[:3]
        attack_feature.append(top_3)

    # attack_feature=P_shadow
    #  label_list = np.expand_dims(label_list, axis=1)
    #  attack_feature=np.concatenate([P_shadow, label_list], axis=1)

    ss = StandardScaler()
    attack_feature = ss.fit_transform(attack_feature)
    return attack_feature


def attack_feature_base2(P_in, P_out, label_list,method):
    assert method in ["CDS", "CT",'DC','SC','DD','SD','ED',"CP", "DF", 'SM','test','LO']

    attack_feature=[]
    if  method == "CT":
        for posterior_in, posterior_out,label in zip(P_in, P_out,label_list):

            attack_feature.append(
                [posterior_in[label], posterior_out[label]])

    elif method == "CDS":
        for posterior_in, posterior_out,label in zip(P_in, P_out,label_list):

            feature_TL_diff=posterior_in[label]-posterior_out[label]
            feature_TL_sum=posterior_in[label]+posterior_out[label]

            attack_feature.append(
                [feature_TL_diff,feature_TL_sum])

    elif method == "DF":
        for posterior_in, posterior_out,label in zip(P_in, P_out,label_list):

            feature_TL_diff=posterior_in[label]-posterior_out[label]
            #
            attack_feature.append(
                [feature_TL_diff])

    elif method == "SM":
        for posterior_in, posterior_out,label in zip(P_in, P_out,label_list):

            feature_TL_sum=posterior_in[label]+posterior_out[label]
            #
            attack_feature.append(
                [feature_TL_sum])

    elif method == "test":
        for posterior_in, posterior_out,label in zip(P_in, P_out,label_list):

            TL_original=posterior_in[label]
            TL_unlearned=posterior_out[label]
            if TL_original<0.5:
                TL_unlearned=0.1
            attack_feature.append(
                [TL_original, TL_unlearned])

    elif method == "DC" or method == "CP":

        return np.concatenate([P_in, P_out], axis=1)

    elif method == "SC":
        P_in = np.sort(P_in, axis=1)
        P_out = np.sort(P_out, axis=1)
        return np.concatenate([P_in, P_out], axis=1)

    elif method == "DD":
        return P_in - P_out

    elif method == "SD":
        P_in = np.sort(P_in, axis=1)
        P_out = np.sort(P_out, axis=1)
        return P_in - P_out

    elif method == "ED":
        return np.sqrt((P_in - P_out) ** 2)

    elif method == "LO":  #publishing label only

        for posterior_in, posterior_out, label in zip(P_in, P_out, label_list):
            # Normalize posterior_in
            max_index_in = np.argmax(posterior_in)
            posterior_in = np.zeros_like(posterior_in)
            posterior_in[max_index_in] = 1

            # Normalize posterior_out
            max_index_out = np.argmax(posterior_out)
            posterior_out = np.zeros_like(posterior_out)
            posterior_out[max_index_out] = 1

            # Extract the value corresponding to the given label
            attack_feature.append([posterior_in[label], posterior_out[label]])
    else:
        raise ValueError("this algorithm is not exist")

    ss = StandardScaler()
    attack_feature = ss.fit_transform(attack_feature)

    return attack_feature

def Double_Attack2(args):


    # 1. Training
    # prep data
    P_shadow_original, P_shadow_unlearned, P_original_target, P_unlearned_target, mem_train, mem_test, train_sample_label, test_sample_label = baseline_prep_for_double_attack(
        args)

    # drop the second dimension
    P_shadow_original = P_shadow_original[:, 0, :]
    P_shadow_unlearned = P_shadow_unlearned[:, 0, :]
    P_original_target = P_original_target[:, 0, :]
    P_unlearned_target = P_unlearned_target[:, 0, :]

    # first attack model
    attack_X = attack_feature_base(P_shadow_original, train_sample_label)
    ytest = mem_train

    Atrain, Aval, y_train, y_val = train_test_split(attack_X, ytest, test_size=0.2, random_state=args['random'])

    clf = MLP2Layer(in_dim=attack_X.shape[1], out_dim=2, layer_list=128, device=torch.device(args['device']))
    clf.criterion = torch.nn.CrossEntropyLoss()
    clf.optimizer = torch.optim.Adam(clf.parameters(), lr=0.001, weight_decay=1e-5)

    clf.to(torch.device(args['device']))

    clf.train_epochs(Atrain, y_train, Aval, y_val, epochs=100, max_patient=20)

    # Testing original model
    attack_X_original_test = attack_feature_base(P_original_target, test_sample_label)

    probs_original = clf.pred(attack_X_original_test)
    y_pred_original = np.argmax(probs_original, axis=1)

    # compare each round
    # precision/recall/f1-score for that class is explicitly 0.0 instead of a warning or NaN.
    value_mapping = {1: 2,
                     2: 1}  # 0 is non-member,1 is member after mapping. 1是原来的目标unlearned sample, 2是retain sample。 在original model中，目标unlearned sample还是retain sample。但还是要保持三个类别
    mem_test_array = np.array([value_mapping.get(x, x) for x in mem_test])

    # 找到mem_test_original中不是2的索引
    mask = mem_test_array != 2
    # 使用布尔索引过滤两个数组
    mem_test_filtered = mem_test_array[mask]

    y_pred_filtered = y_pred_original[mask]
    acc_original = class_specific_accuracy(mem_test_filtered, y_pred_filtered)
    f1_original = class_specific_f1(mem_test_filtered, y_pred_filtered)
    precision_original = class_specific_precision(mem_test_filtered, y_pred_filtered)

    # second attack model
    attack_X = attack_feature_base(P_shadow_unlearned, train_sample_label)
    ytest = mem_train
    Atrain, Aval, y_train, y_val = train_test_split(attack_X, ytest, test_size=0.2, random_state=args['random'])

    clf = MLP2Layer(in_dim=attack_X.shape[1], out_dim=2, layer_list=128, device=torch.device(args['device']))

    clf.criterion = torch.nn.CrossEntropyLoss()
    clf.optimizer = torch.optim.Adam(clf.parameters(), lr=0.001, weight_decay=1e-5)

    clf.to(torch.device(args['device']))

    clf.train_epochs(Atrain, y_train, Aval, y_val, epochs=100, max_patient=20)

    # Testing unlearned model
    attack_X_unlearned_test2 = attack_feature_base(P_unlearned_target, test_sample_label)
    probs_unlearned = clf.pred(attack_X_unlearned_test2)
    y_pred_unlearned = np.argmax(probs_unlearned, axis=1)

    # second round focus on member , label 1
    y_pred_filtered = y_pred_unlearned[mask]
    acc_unlearned = class_specific_accuracy(mem_test_filtered, y_pred_filtered)
    f1_unlearned = class_specific_f1(mem_test_filtered, y_pred_filtered)
    precision_unlearned = class_specific_precision(mem_test_filtered, y_pred_filtered)

    # 保存8个指标值
    save_path = os.getcwd() + f"/result/{args['attack_method']}/{args['U_method']}/{args['net_name']}/{args['dataset_name']}/{args['proportion_of_group_unlearn']}/"
    os.makedirs(save_path, exist_ok=True)



    # ========== 结合两个模型进行攻击 ==========
    print("\n========== 结合两个模型进行攻击 (Double Attack) ==========")

    Pin, Pout, Pin_test, Pout_test, mem_train, mem_test, train_sample_label, test_sample_label = baseline_prep_for_double_attack2(args)


    # drop the second dimension
    Pin = Pin[:, 0, :]
    Pout = Pout[:, 0, :]
    Pin_test = Pin_test[:, 0, :]
    Pout_test = Pout_test[:, 0, :]



    method= 'CT'
    attack_X = attack_feature_base2(Pin, Pout, train_sample_label, method)
    ytest = mem_train

    Atrain, Aval, y_train, y_val = train_test_split(attack_X, ytest, test_size=0.2, random_state=args['random'])

    clf = MLP2Layer(in_dim=attack_X.shape[1], out_dim=3, layer_list=128,
                    device=torch.device(args['device']))

    clf.criterion = torch.nn.CrossEntropyLoss()
    clf.optimizer = torch.optim.Adam(clf.parameters(), lr=0.001, weight_decay=1e-5)

    clf.to(torch.device(args['device']))

    clf.train_epochs(Atrain, y_train, Aval, y_val, epochs=500, max_patient=200)

    # 2. Testing
    attack_X_test = attack_feature_base2(Pin_test, Pout_test, test_sample_label, method)

    probs = clf.pred(attack_X_test)
    y_pred = np.argmax(probs, axis=1)
    y_target = mem_test


    y_target = np.array(y_target)
    y_pred = np.array(y_pred)
    
    # 计算 retain set 的准确率：在 retain set 中，有多少被正确识别为 retain (y_pred == 2)
    acc = np.mean(y_pred[y_target == 2] == 2)
    
    # 计算 double attack 的 retain set precision 和 F1-score
    precision_double = class_specific_precision(y_target, y_pred, classes=[2])
    f1_double = class_specific_f1(y_target, y_pred, classes=[2])
    metrics_dict = {
        "original_model_retain_acc": round(acc_original[1], 4),
        "unlearned_model_retain_acc": round(acc_unlearned[1], 4),
        "double_attack_retain_acc": round(acc, 4),
        "original_model_retain_precision": round(precision_original[1], 4),
        "unlearned_model_retain_precision": round(precision_unlearned[1], 4),
        "double_attack_retain_precision": round(precision_double[0], 4),
        "original_model_retain_f1": round(f1_original[1], 4),
        "unlearned_model_retain_f1": round(f1_unlearned[1], 4),
        "double_attack_retain_f1": round(f1_double[0], 4),
    }
    # 创建 DataFrame，转置存储 (指标名 | 对应值)
    data_to_save = pd.DataFrame(list(metrics_dict.items()), columns=["Metric", "Value"])
    # 保存为 CSV 文件
    data_to_save.to_csv(f"{save_path}/class_metrics.csv", index=False)

    print(f"指标已保存到: {save_path}/class_metrics.csv")

def class_specific_accuracy(y_true, y_pred, classes=[0, 1]):
    acc_list = []
    for class_label in classes:
        # 对于当前类别，正确预测为该类别的数量 / 实际为该类别的数量
        true_positives = np.sum((y_true == class_label) & (y_pred == class_label))
        total_actual = np.sum(y_true == class_label)
        class_acc = true_positives / total_actual if total_actual > 0 else 0
        acc_list.append(class_acc)
    return acc_list


def class_specific_precision(y_true, y_pred, classes=[0, 1]):
    """计算每个类别的precision（精确率）"""
    precision_list = []
    for class_label in classes:
        # 计算当前类别的precision
        true_positives = np.sum((y_true == class_label) & (y_pred == class_label))
        false_positives = np.sum((y_true != class_label) & (y_pred == class_label))
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        precision_list.append(precision)
    return precision_list


def class_specific_f1(y_true, y_pred, classes=[0, 1]):
    f1_list = []
    for class_label in classes:
        # 计算当前类别的F1-score
        true_positives = np.sum((y_true == class_label) & (y_pred == class_label))
        false_positives = np.sum((y_true != class_label) & (y_pred == class_label))
        false_negatives = np.sum((y_true == class_label) & (y_pred != class_label))

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_list.append(f1)
    return f1_list
