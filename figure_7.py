from collections import defaultdict

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.pyplot import ylabel
from scipy.special import softmax
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from attack.metric.metric import all_metrics_for_three_class
from parameter_parser import parameter_parser
from data.load_data import get_data
from data.prepare_data import construct_dataset, split_dataset
import os
from model.DNN import DNN
import pandas as pd
import scipy
import matplotlib.pyplot as plt

class MLP2Layer(nn.Module):
    def __init__(self, in_dim, out_dim, layer_list, device):
        super(MLP2Layer, self).__init__()
        assert len(layer_list) == 2
        torch.manual_seed(3407)

        self.fc1 = nn.Linear(in_dim, layer_list[0])
        self.fc2 = nn.Linear(layer_list[0], layer_list[1])
        self.fc3 = nn.Linear(layer_list[1], out_dim)

        self.outdim = out_dim
        self.indim = in_dim

        self.criterion = None
        self.optimizer = None
        self.device = device

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
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


    def all_metrics(self, X_target, y_target):

        outputs_target = self(torch.Tensor(X_target).to(self.device)).cpu()

        acc_target = metrics.accuracy_score(y_target, outputs_target.detach().numpy().argmax(axis=1))
        prec_target = metrics.precision_score(y_target, outputs_target.detach().numpy().argmax(axis=1))
        recall_target = metrics.recall_score(y_target, outputs_target.detach().numpy().argmax(axis=1))
        auc_target = metrics.roc_auc_score(y_target, outputs_target.detach().numpy()[:, 1])
        f1_target = metrics.f1_score(y_target, outputs_target.detach().numpy().argmax(axis=1))

        tpr_05fpr = tpr_at_fpr(y_target, outputs_target.detach().numpy()[:, 1], 0.05)
        tpr_10fpr = tpr_at_fpr(y_target, outputs_target.detach().numpy()[:, 1], 0.10)
        tpr_001fpr = tpr_at_fpr(y_target, outputs_target.detach().numpy()[:, 1], 0.01)
        return [acc_target, prec_target, recall_target, auc_target, f1_target,
                tpr_001fpr, tpr_05fpr, tpr_10fpr]


def tpr_at_fpr(y_true, y_score, fpr_th):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    ind = np.argmin(np.abs(fpr - fpr_th))
    tpr_res = max(tpr[fpr==fpr[ind]])
    return tpr_res

def attack_feature_base(P_in, P_out, label_list,method):


    attack_feature=[]
    if method == "Concate":
        for posterior_in, posterior_out,label in zip(P_in, P_out,label_list):

            attack_feature.append(
                [posterior_in[label], posterior_out[label]])

    elif method == "SaD":
        for posterior_in, posterior_out,label in zip(P_in, P_out,label_list):

            feature_TL_diff=posterior_in[label]-posterior_out[label]
            feature_TL_sum=posterior_in[label]+posterior_out[label]
            #
            attack_feature.append(
                [feature_TL_diff,feature_TL_sum])

    elif method == "Sum":
        for posterior_in, posterior_out,label in zip(P_in, P_out,label_list):

            feature_TL_sum=posterior_in[label]+posterior_out[label]
            #
            attack_feature.append(
                [feature_TL_sum])

    elif method == "Diff":
        for posterior_in, posterior_out,label in zip(P_in, P_out,label_list):

            feature_TL_diff=posterior_in[label]-posterior_out[label]
            attack_feature.append(
                [feature_TL_diff])

    elif method == "test":
        for posterior_in, posterior_out,label in zip(P_in, P_out,label_list):

            TL_original=posterior_in[label]
            TL_unlearned=posterior_out[label]
            if TL_original<0.5:
                TL_unlearned=0.1
            attack_feature.append(
                [TL_original, TL_unlearned])

    elif method == "DC":

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
    else:
        raise ValueError("this algorithm is not exist")


    return attack_feature

def attack_feature_base4(P_in, P_out, label_list,method):
    attack_feature = []
    o_output =  []
    u_output=[]
    for posterior_in, posterior_out, label in zip(P_in, P_out, label_list):
        o_max_index = np.argmax(posterior_in)
        o_output.append(int(o_max_index == label))
        u_max_index = np.argmax(posterior_out)
        u_output.append(int(u_max_index == label))

    return  o_output,u_output
def attack_feature_base2(P_in, P_out, label_list,method):
    attack_feature = []
    #attack_feature= np.concatenate([P_in, P_out], axis=1)

    for posterior_in, posterior_out, label in zip(P_in, P_out, label_list):
        feature_max_diff = max(posterior_in) - max(posterior_out)
        feature_max_sum = max(posterior_in) + max(posterior_out)

        feature_TL_diff = posterior_in[label] - posterior_out[label]
        feature_TL_sum = posterior_in[label] + posterior_out[label]

        feature_Var_diff = np.var(posterior_in) - np.var(posterior_out)
        feature_Var_sum = np.var(posterior_in) + np.var(posterior_out)

        posterior_in = sorted(posterior_in, reverse=True)
        posterior_out = sorted(posterior_out, reverse=True)
        confidence_feature_in = posterior_in[0] - posterior_in[1]
        confidence_feature_out = posterior_out[0] - posterior_out[1]
        feature_Conf_diff = confidence_feature_in - confidence_feature_out
        feature_Conf_sum = confidence_feature_out + confidence_feature_out

        # attack_feature.append(
        #     [feature_TL_diff, feature_TL_sum, feature_Var_diff, feature_Var_sum, feature_Conf_diff, feature_Conf_sum,
        #      feature_max_diff, feature_max_sum])
       # attack_feature.append([max(posterior_in),max(posterior_out)])
        attack_feature.append([feature_Conf_diff,feature_Conf_sum])

    ss = StandardScaler()
    attack_feature = ss.fit_transform(attack_feature)
    return attack_feature


def probability_bar(A,B,method):
    # 计算和
    sum_A = np.array(A)
    sum_B = np.array(B)

    # 设定 bins (分区范围)
    if method=='Sum':
        bins = np.linspace(0, 2, 10)
    elif method=='Diff':
        bins = np.linspace(-1, 1, 10)

    # 设置图像大小

    plt.figure(figsize=(8, 6))

    # 计算每个 bin 内的数据数量
    counts_A, bin_edges = np.histogram(sum_A, bins=bins)
    counts_B, _ = np.histogram(sum_B, bins=bins)

    # 计算百分比
    total_A = sum(counts_A)
    total_B = sum(counts_B)

    percent_A = counts_A / total_A * 100
    percent_B = counts_B / total_B * 100

    # 绘制直方图（用 bar 替代 hist 以控制 y 轴）
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.bar(bin_centers, percent_A, width=(bin_edges[1] - bin_edges[0]) * 0.4, alpha=0.7, color='#007acc',
            label='Retrain', edgecolor='black')
    plt.bar(bin_centers + (bin_edges[1] - bin_edges[0]) * 0.4, percent_B, width=(bin_edges[1] - bin_edges[0]) * 0.4,
            alpha=0.7, color='#ff6600', label='SISA', edgecolor='black')

    # 设置 y 轴显示百分比
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))

    # 设置标题和坐标轴标签
    plt.xlabel("Range of Probabilities", fontsize=22)
    plt.ylabel("Percentage of Samples", fontsize=22)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # 添加图例，放在右上角
    plt.legend(loc="best", fontsize=22, frameon=True)

    # 显示网格线，增强可读性
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.subplots_adjust(left=0.15)  # 适当增大 left 参数，使图像右移

    # 显示图像
    plt.show()


def probability_bar2(A,B,method):
    # 计算和
    sum_A = np.array(A)
    sum_B = np.array(B)

    # 设定 bins (分区范围)
    if method=='retrain':
        bins = np.linspace(0, 1, 10)
    elif method=='SISA':
        bins = np.linspace(-1, 1, 10)
    bins = np.linspace(0, 1, 10)

    # 设置图像大小

    plt.figure(figsize=(8, 6))

    # 计算每个 bin 内的数据数量
    counts_A, bin_edges = np.histogram(sum_A, bins=bins)
    counts_B, _ = np.histogram(sum_B, bins=bins)

    # 计算百分比
    total_A = sum(counts_A)
    total_B = sum(counts_B)

    percent_A = counts_A / total_A * 100
    percent_B = counts_B / total_B * 100

    # 绘制直方图（用 bar 替代 hist 以控制 y 轴）
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.bar(bin_centers, percent_A, width=(bin_edges[1] - bin_edges[0]) * 0.4, alpha=0.7, color='#007acc',
            label='Before Unlearning', edgecolor='black')
    plt.bar(bin_centers + (bin_edges[1] - bin_edges[0]) * 0.4, percent_B, width=(bin_edges[1] - bin_edges[0]) * 0.4,
            alpha=0.7, color='#ff6600', label='After Unlearning', edgecolor='black')

    # 设置 y 轴显示百分比
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))

    # 设置标题和坐标轴标签
    plt.xlabel("Range of Probabilities", fontsize=22)
    plt.ylabel("Percentage of Samples", fontsize=22)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # 添加图例，放在右上角
    plt.legend(loc="best", fontsize=22, frameon=True)

    # 显示网格线，增强可读性
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.subplots_adjust(left=0.15)  # 适当增大 left 参数，使图像右移

    # 显示图像
    plt.show()


# Retrain and SISA in one image
def deal_data(sample,Pin_test,Pout_test,test_sample_label,mem_test,Pin_test_sisa,Pout_test_sisa,test_sample_label_sisa,mem_test_sisa,method):
    attack_X = attack_feature_base(Pin_test, Pout_test, test_sample_label, method)

    attack_X_sisa = attack_feature_base(Pin_test_sisa, Pout_test_sisa, test_sample_label_sisa, method)

    if sample=='unseen':
        attack_X = [a for a, b in zip(attack_X, mem_test) if b == 0]
        attack_X_sisa = [a for a, b in zip(attack_X_sisa, mem_test_sisa) if b == 0]
        if method=='Sum':
            attack_X = [x for x in attack_X if x[0] <= 1.85]
            attack_X_sisa = [x for x in attack_X_sisa if x[0] <= 1.8]


    elif sample=='forget':
        attack_X = [a for a, b in zip(attack_X, mem_test) if b == 1]
        attack_X_sisa = [a for a, b in zip(attack_X_sisa, mem_test_sisa) if b == 1]
        if method=='Sum':
            attack_X = [x for x in attack_X if x[0] <= 1.85]
            attack_X_sisa = [x for x in attack_X_sisa if x[0] <= 1.8]
    else:
        attack_X = [a for a, b in zip(attack_X, mem_test) if b == 2]
        attack_X_sisa = [a for a, b in zip(attack_X_sisa, mem_test_sisa) if b == 2]

    probability_bar(attack_X,attack_X_sisa,method)

# before and after in one image.
def deal_data2(sample,Pin_test,Pout_test,test_sample_label,mem_test,Pin_test_sisa,Pout_test_sisa,test_sample_label_sisa,mem_test_sisa,method):
    Pin_test_list=[]
    Pout_test_list=[]
    Pin_test_sisa_list=[]
    Pout_test_sisa_list=[]
    for posterior_in, posterior_out, label in zip(Pin_test, Pout_test, test_sample_label):
        Pin_test_list.append([posterior_in[label]])
        Pout_test_list.append([posterior_out[label]])

    for posterior_in, posterior_out, label in zip(Pin_test_sisa, Pout_test_sisa, test_sample_label_sisa):
        Pin_test_sisa_list.append([posterior_in[label]])
        Pout_test_sisa_list.append([posterior_out[label]])


    if method=='retrain':
        if sample=='unseen':
            attack_X_before = [a for a, b in zip(Pin_test_list, mem_test) if b == 0]
            attack_X_after = [a for a, b in zip(Pout_test_list, mem_test) if b == 0]
           # attack_X_before = [x for x in attack_X_before if x[0] <= 0.9]


        elif sample=='forget':
            attack_X_before = [a for a, b in zip(Pin_test_list, mem_test) if b == 1]
            attack_X_after = [a for a, b in zip(Pout_test_list, mem_test) if b == 1]

        else:
            attack_X_before = [a for a, b in zip(Pin_test_list, mem_test) if b == 2]
            attack_X_after = [a for a, b in zip(Pout_test_list, mem_test) if b == 2]
    else:
        if sample == 'unseen':
            attack_X_before = [a for a, b in zip(Pin_test_sisa_list, mem_test) if b == 0]
            attack_X_after = [a for a, b in zip(Pout_test_sisa_list, mem_test) if b == 0]
            attack_X_before = [x for x in attack_X_before if x[0] <= 0.995]


        elif sample == 'forget':
            attack_X_before = [a for a, b in zip(Pin_test_sisa_list, mem_test) if b == 1]
            attack_X_after = [a for a, b in zip(Pout_test_sisa_list, mem_test) if b == 1]

        else:
            attack_X_before = [a for a, b in zip(Pin_test_sisa_list, mem_test) if b == 2]
            attack_X_after = [a for a, b in zip(Pout_test_sisa_list, mem_test) if b == 2]
            attack_X_after = [x for x in attack_X_after if x[0] >= 0.4]

    probability_bar2(attack_X_before,attack_X_after,method)



def Image_Printing(args,sample,method):
    print(f"{args['U_method']}-{args['net_name']}-{args['dataset_name']}-----")
    # prep data
    args['U_method'] ='retrain'

    Pin, Pout, Pin_test, Pout_test, mem_train, mem_test, train_sample_label, test_sample_label = baseline_prep(args)
    # drop the second dimension
    args['U_method'] ='sparsity'
    Pin_sisa, Pout_sisa, Pin_test_sisa, Pout_test_sisa, mem_train_sisa, mem_test_sisa, train_sample_label_sisa, test_sample_label_sisa = baseline_prep(args)

    Pin_test = Pin_test[:, 0, :]
    Pout_test = Pout_test[:, 0, :]

    Pin_test_sisa = Pin_test_sisa[:, 0, :]
    Pout_test_sisa = Pout_test_sisa[:, 0, :]

    # for method in ["TL", "Max", "Conf","Var", "All"]:
    #  for method in ["SaD", "Concate", 'diff', 'sum', 'All']:
  #  deal_data(sample,Pin_test, Pout_test, test_sample_label, mem_test, Pin_test_sisa, Pout_test_sisa,test_sample_label_sisa, mem_test_sisa, method)

    deal_data2(sample,Pin_test, Pout_test, test_sample_label, mem_test, Pin_test_sisa, Pout_test_sisa,test_sample_label_sisa, mem_test_sisa, method)



def percent(data_retrain,data_sisa,classes):
    data_retrain=data_retrain.flatten()
    data_sisa=data_sisa.flatten()

    # 1. 排序数据
    sorted_data_retrain = np.sort(data_retrain)
    sorted_data_sisa = np.sort(data_sisa)

    # 2. 计算累积分布（百分比）
    percent_retrain = np.linspace(0, 100, len(sorted_data_retrain))
    percent_sisa = np.linspace(0, 100, len(sorted_data_sisa))

    # 3. 绘制 CDF
    plt.figure(figsize=(6, 4))
    plt.plot(sorted_data_retrain, percent_retrain, label="Retrain", color='blue')
    plt.plot(sorted_data_sisa, percent_sisa, label="SISA", color='red')
    plt.legend(fontsize=14)
    plt.subplots_adjust(bottom=0.2)  # 调整底部间距，增加数值让图像上移

    # 4. 添加标签和标题
    plt.xlabel("(Predicted membership probability after unlearning) - \n"
               "(Predicted membership probability before unlearning)",fontsize=12)
    plt.ylabel("Percent",fontsize=18)
   # plt.title("Cumulative Distribution Function (CDF)")
    plt.grid(True)

    # 5. 显示图像
    plt.show()




def baseline_prep(args):

    P_original_model_target = []
    P_unlearned_model_target = []
    target_class = []
    P_original_model_shadow = []
    P_unlearned_model_shadow = []
    shadow_class = []
    shadow_sample_label=[]
    target_sample_label=[]

    target_save_path = os.getcwd() + f"/save/{args['U_method']}/{args['net_name']}/{args['dataset_name']}/{args['proportion_of_group_unlearn']}/target/0"
   # target_shadow_path = os.getcwd() + f"/save/{args['U_method']}/{args['net_name']}/{args['dataset_name']}/0.02/shadow/0"


    target_num_subfolders=sum(os.path.isdir(os.path.join(target_save_path, name)) for name in os.listdir(target_save_path))

    # target
    for t in tqdm(range(args['trials']), desc='preparing members'):

        for i in range(target_num_subfolders):
            if args['flag']=='none':
                save_path = os.getcwd() + f"/save/{args['U_method']}/{args['net_name']}/{args['dataset_name']}/{args['proportion_of_group_unlearn']}/target/{t}/{i}"
            else:
                # different type samples
                save_path = os.getcwd() + f"/save/{args['U_method']}/{args['net_name']}/{args['dataset_name']}/{args['proportion_of_group_unlearn']}/{args['flag']}/{t}/{i}"
         #   save_path = os.getcwd() + f"/save/{args['U_method']}/{args['net_name']}/{args['dataset_name']}/{args['proportion_of_group_unlearn']}/{args['sigma']}/target/{t}/{i}"
         #   save_path = os.getcwd() + f"/save/sparsity/{args['net_name']}/{args['dataset_name']}/{args['proportion_of_group_unlearn']}/target/{t}/{i}"
          #  save_path = os.getcwd() + f"/save/{args['U_method']}/{args['net_name']}/{args['dataset_name']}/{args['proportion_of_group_unlearn']}/over-well/{t}/{i}"

            #  impact of various parameters
         #   save_path = os.getcwd() + f"/save/{args['U_method']}/{args['net_name']}/{args['dataset_name']}/{args['proportion_of_group_unlearn']}/well-well/{t}/{i}"

            # unlearned_sample_posterior
            unlearned_sample_original_model_posterior = torch.load(
                f"{save_path}/unlearned_sample_original_model_posterior.pth")
            unlearned_sample_unlearned_model_posterior = torch.load(
                f"{save_path}/unlearned_sample_unlearned_model_posterior.pth")
            unlearned_sample_label=torch.load(f"{save_path}/target_sample_label.pth")

            P_original_model_target.append(unlearned_sample_original_model_posterior)
            P_unlearned_model_target.append(unlearned_sample_unlearned_model_posterior)
            target_class.append(1)
            target_sample_label.append(unlearned_sample_label)

            # unseen_sample_posterior
            unseen_sample_original_model_posterior = torch.load(f"{save_path}/unseen_sample_original_model_posterior.pth")
            unseen_sample_unlearned_model_posterior = torch.load(f"{save_path}/unseen_sample_unlearned_model_posterior.pth")
            unseen_sample_label=torch.load(f"{save_path}/unseen_sample_label.pth")

            P_original_model_target.append(unseen_sample_original_model_posterior)
            P_unlearned_model_target.append(unseen_sample_unlearned_model_posterior)
            target_class.append(0)
            target_sample_label.append(unseen_sample_label)

            # retain_sample_posterior
            retain_sample_original_model_posterior = torch.load(f"{save_path}/retain_sample_original_model_posterior.pth")
            retain_sample_unlearned_model_posterior = torch.load(f"{save_path}/retain_sample_unlearned_model_posterior.pth")
            retain_sample_label=torch.load(f"{save_path}/retain_sample_label.pth")

            P_original_model_target.append(retain_sample_original_model_posterior)
            P_unlearned_model_target.append(retain_sample_unlearned_model_posterior)
            target_class.append(2)
            target_sample_label.append(retain_sample_label)

    P_original_model_shadow = np.array(P_original_model_shadow)
    P_unlearned_model_shadow = np.array(P_unlearned_model_shadow)
    P_original_model_target = np.array(P_original_model_target)
    P_unlearned_model_target = np.array(P_unlearned_model_target)
    return P_original_model_shadow, P_unlearned_model_shadow, P_original_model_target, P_unlearned_model_target, shadow_class, target_class,shadow_sample_label,target_sample_label

if __name__ == '__main__':
    args = parameter_parser()
    args['dataset_name'] ='cifar100'
    args['proportion_of_group_unlearn'] =0.02
    args['net_name'] ='resnet18'
    args['num_epochs'] =50
    args['U_method'] ='retrain'

    args['observations'] =1
    args['trials'] =1
    sample='retain'
    method='Sum'  # ['Diff','Sum','retrain','sisa']
    Image_Printing(args,sample, method)  # for figure 7
