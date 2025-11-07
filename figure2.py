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


import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def t_SNE(A):
    # 转换为 numpy 数组
    A_array = np.array(A)

    # 使用 t-SNE 降维到 1 维
    # tsne = TSNE(n_components=1, perplexity=50,
    # n_iter=2000,            # 增加迭代次数
    # learning_rate=200,      # 可以尝试150-300
    # early_exaggeration=12, random_state=42)
    tsne = TSNE(n_components=1, random_state=42)
    A_tsne = tsne.fit_transform(A_array)

    # 将结果展平为一维数组
    A_flat = A_tsne.flatten()
    return A_flat
def vis(A,B):


    # Flatten list A for easier processing (A contains nested lists)
    A_flat = [x[0] for x in A]
    A_flat=t_SNE(A)
    colors = {0: 'blue', 1: 'green', 2: 'red'}
    labels = {0: 'unseen set', 1: 'forget set', 2: 'retain set'}
    C = np.random.uniform(0, 1, len(B))  # 余弦相似度

    # Create the plot
    plt.figure(figsize=(8, 6))

    # Plot points for each category
    for b_value, color in colors.items():
        indices = [i for i, b in enumerate(B) if b == b_value]
        plt.scatter([C[i] for i in indices],
                    [A_flat[i] for i in indices],
                    color=color,
                    label=labels[b_value],
                    alpha=0.8)

    # Add labels and title
  #  plt.title("sum", fontsize=18)
   # plt.xlabel("features", fontsize=18)
    plt.tick_params(axis='y', labelsize=14)  # 放大刻度字体
    plt.xticks([])  # 隐藏横坐标刻度
    plt.ylabel("Embedding Values", fontsize=22)
    plt.grid(True, linestyle='--', alpha=0.7)
  #  plt.legend(fontsize=22,loc='lower left')

    # Show the plot
    plt.show()
# def compute_acc_class(o_pred,u_pred,classes):
#     # 初始化计数字典
#     correct = defaultdict(int)
#     total = defaultdict(int)
#
#     # 统计每个类别的正确样本数和总样本数
#     for pred, label in zip(A, B):
#         total[label] += 1
#         correct[label] += pred  # 1 表示正确
#
#     # 计算正确率
#     accuracy = {label: correct[label] / total[label] for label in total}
#     return accuracy
def test():
    # 生成示例数据
    np.random.seed(42)
    num_samples = 100
    cosine_similarities = np.random.uniform(0, 1, num_samples)  # 余弦相似度
    ground_truth_labels = np.random.choice([0, 1], num_samples)  # 真实标签（0 或 1）

    # 创建散点图
    plt.figure(figsize=(8, 6))
    plt.scatter(cosine_similarities, ground_truth_labels, alpha=0.6, edgecolors='w', s=100)

    # 设置标题和标签
    plt.title('Cosine Similarity vs Ground-Truth Membership Labels', fontsize=14)
    plt.xlabel('Cosine Similarity', fontsize=12)
    plt.ylabel('Ground-Truth Membership Labels', fontsize=12)

    # 设置 y 轴刻度
    plt.yticks([0, 1], ['Non-Member', 'Member'])

    # 显示网格
    plt.grid(True, linestyle='--', alpha=0.6)

    # 显示图形
    plt.show()



def baseline_prep(args):
    P_original_model_target = []
    P_unlearned_model_target = []
    target_class = []
    P_original_model_shadow = []
    P_unlearned_model_shadow = []
    shadow_class = []
    shadow_sample_label = []
    target_sample_label = []

    target_save_path = os.getcwd() + f"/save/{args['U_method']}/{args['net_name']}/{args['dataset_name']}/{args['proportion_of_group_unlearn']}/target/0"

    if args['flag'] != 'none':
        target_save_path = os.getcwd() + f"/save/{args['U_method']}/{args['net_name']}/{args['dataset_name']}/{args['proportion_of_group_unlearn']}/{args['flag']}/0"

    target_num_subfolders = sum(
        os.path.isdir(os.path.join(target_save_path, name)) for name in os.listdir(target_save_path))
    # target
    target_num_subfolders=192
    for t in tqdm(range(args['trials']), desc='preparing members'):

        for i in range(target_num_subfolders):
            if args['flag'] == 'none':
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
            unlearned_sample_label = torch.load(f"{save_path}/target_sample_label.pth")

            P_original_model_target.append(unlearned_sample_original_model_posterior)
            P_unlearned_model_target.append(unlearned_sample_unlearned_model_posterior)
            target_class.append(1)
            target_sample_label.append(unlearned_sample_label)

            # unseen_sample_posterior
            unseen_sample_original_model_posterior = torch.load(
                f"{save_path}/unseen_sample_original_model_posterior.pth")
            unseen_sample_unlearned_model_posterior = torch.load(
                f"{save_path}/unseen_sample_unlearned_model_posterior.pth")
            unseen_sample_label = torch.load(f"{save_path}/unseen_sample_label.pth")

            P_original_model_target.append(unseen_sample_original_model_posterior)
            P_unlearned_model_target.append(unseen_sample_unlearned_model_posterior)
            target_class.append(0)
            target_sample_label.append(unseen_sample_label)

            # retain_sample_posterior
            retain_sample_original_model_posterior = torch.load(
                f"{save_path}/retain_sample_original_model_posterior.pth")
            retain_sample_unlearned_model_posterior = torch.load(
                f"{save_path}/retain_sample_unlearned_model_posterior.pth")
            retain_sample_label = torch.load(f"{save_path}/retain_sample_label.pth")

            P_original_model_target.append(retain_sample_original_model_posterior)
            P_unlearned_model_target.append(retain_sample_unlearned_model_posterior)
            target_class.append(2)
            target_sample_label.append(retain_sample_label)



    P_original_model_shadow = np.array(P_original_model_shadow)
    P_unlearned_model_shadow = np.array(P_unlearned_model_shadow)
    P_original_model_target = np.array(P_original_model_target)
    P_unlearned_model_target = np.array(P_unlearned_model_target)
    return P_original_model_shadow, P_unlearned_model_shadow, P_original_model_target, P_unlearned_model_target, shadow_class, target_class, shadow_sample_label, target_sample_label

if __name__ == '__main__':
    args = parameter_parser()
    args['dataset_name'] ='cifar100'   #I use tinyimagenet when SaD
    args['proportion_of_group_unlearn'] =0.02
    args['net_name'] ='resnet18'
    args['num_epochs'] =50
    args['U_method'] ='retrain'

    args['observations'] =1
    args['trials'] =1
    method='SaD'  # ['Diff','Sum','retrain','sisa']
    _, _, Pin_test, Pout_test, mem_train, mem_test, train_sample_label, test_sample_label = baseline_prep(args)

    Pin_test = Pin_test[:, 0, :]
    Pout_test = Pout_test[:, 0, :]


    attack_X = attack_feature_base(Pin_test, Pout_test, test_sample_label, method)


    vis(attack_X,mem_test)