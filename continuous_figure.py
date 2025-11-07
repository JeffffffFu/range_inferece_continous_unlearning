from collections import defaultdict

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.pyplot import ylabel
from scipy.special import softmax
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
from sympy.abc import epsilon
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
import ruptures as rpt






def baseline_prep(args,K):

    # 每个 (t, k) 下的样本子目录数量将动态读取保存目录确定
    model_index=4
    confident_retain_list=[]
    confident_forget_list=[]
    confident_unseen_list=[]

    for k in range(K):
        sum_conf_forget = 0.0
        sum_conf_retain = 0.0
        sum_conf_unseen = 0.0
        count = 0  # 统计累加的样本总数（trials × 子目录数量）

        for t in tqdm(range(args['trials']), desc=f'collect@step{k}'):
            # 基础路径（不含样本子目录 i）
            base_dir = os.getcwd() + f"/save/{args['U_method']}/{args['net_name']}/{args['dataset_name']}/{args['proportion_of_group_unlearn']}/tracked_forget/track_from_{model_index}/{t}/{k}"
            base_dir = os.getcwd() + f"/save/{args['U_method']}/{args['net_name']}/{args['dataset_name']}/{args['proportion_of_group_unlearn']}/tracked_data_A/{t}/{k}"

            # 统计该 (t, k) 下的样本子目录数量
            subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
            for i in range(len(subdirs)):
                save_path = f"{base_dir}/{i}"
                unlearned_post = torch.load(f"{save_path}/unlearned_sample_unlearned_model_posterior.pth", weights_only=False)
             #   unlearned_label = torch.load(f"{save_path}/target_sample_label.pth", weights_only=False)
              #  conf_forget = float(unlearned_post[0][int(unlearned_label)])
                conf_forget = float(max(unlearned_post[0]))

                unseen_post= torch.load(f"{save_path}/unseen_sample_unlearned_model_posterior.pth", weights_only=False)
             #   unseen_label = torch.load(f"{save_path}/unseen_sample_label.pth", weights_only=False)
             #   conf_unseen = float(unseen_post[0][int(unseen_label)])
                conf_unseen = float(max(unseen_post[0]))

                retain_post_u = torch.load(f"{save_path}/retain_sample_unlearned_model_posterior.pth", weights_only=False)
             #   retain_label = torch.load(f"{save_path}/retain_sample_label.pth", weights_only=False)
              #  conf_retain = float(retain_post_u[0][int(retain_label)])
                conf_retain = float(max(retain_post_u[0]))


                sum_conf_forget += conf_forget
                sum_conf_unseen += conf_unseen
                sum_conf_retain += conf_retain
                count += 1

        confident_forget_list.append(sum_conf_forget / count if count>0 else 0.0)
        confident_retain_list.append(sum_conf_retain / count if count>0 else 0.0)
        confident_unseen_list.append(sum_conf_unseen / count if count>0 else 0.0)


    return confident_retain_list, confident_forget_list, confident_unseen_list

def detect_and_plot_changepoints(values_list, n_bkps=2, model="kernel"):

    model='kernel'
    if not isinstance(values_list, (list, tuple)) or len(values_list) == 0:
        raise ValueError("values_list 必须是非空的 list/tuple")

    signal = np.asarray(values_list, dtype=float).reshape(-1, 1)

    # 使用可指定变点数量的算法：Binseg；若为 kernel 则使用 KernelCPD
    if model == "kernel":
        algo = rpt.KernelCPD(kernel="linear").fit(signal)
        bkps = algo.predict(n_bkps=n_bkps)

    else:
        algo = rpt.Binseg(model=model).fit(signal)
       # bkps = algo.predict(n_bkps=n_bkps)
        bkps = algo.predict(epsilon=0.001)

    plt.figure(figsize=(12, 4))
    plt.plot(values_list, color='k', linewidth=1.5, label='Raw data')

    # 绘制分段常数拟合线（蓝色）并在断点处用竖线连接
    segment_means = []
    start = 0
    for cp in bkps:
        end = cp
        segment_mean = float(np.mean(signal[start:end])) if end > start else float(signal[start])
        segment_means.append(segment_mean)
        # 横线段（右端点不包含，因此减 1 更贴合）
        if end == len(values_list):
            end=len(values_list)-1
        plt.hlines(segment_mean, start, max(end, start), colors='royalblue', linewidth=3)
        start = end

    # 在每个断点位置画竖向连接线，将相邻两段的均值连起来
    for idx in range(len(bkps) - 1):
        x = bkps[idx]
        y1 = segment_means[idx]
        y2 = segment_means[idx + 1]
        plt.vlines(x, min(y1, y2), max(y1, y2), colors='royalblue', linewidth=3)
    
    # 添加CUSUM process标签（只显示一次）
    plt.plot([], [], color='royalblue', linewidth=3, label='CUSUM process')

    # 设置 x 轴刻度为 1..N
    x_ticks = range(1, len(values_list) + 1)
    plt.xticks(ticks=range(len(values_list)), labels=x_ticks)
    plt.xlabel('Timestamp',fontsize=10)
    plt.ylabel('Avg confidence score',fontsize=12)
  #  plt.title('Change point detection')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

    return bkps

def detect_and_plot_dual_changepoints(values_list1, values_list2, n_bkps=2, model="kernel", labels=None):
    """
    检测并绘制两个时间序列的变点
    
    Args:
        values_list1: 第一个时间序列
        values_list2: 第二个时间序列
        n_bkps: 期望的断点数量
        model: 变点检测模型 ("kernel" 或 "l2")
        labels: 两个序列的标签，默认为 ["Series 1", "Series 2"]
    """
    if labels is None:
        labels = ["Forget set", "Unseen set"]
    
    if not isinstance(values_list1, (list, tuple)) or len(values_list1) == 0:
        raise ValueError("values_list1 必须是非空的 list/tuple")
    if not isinstance(values_list2, (list, tuple)) or len(values_list2) == 0:
        raise ValueError("values_list2 必须是非空的 list/tuple")
    
    # 确保两个序列长度相同
    min_len = min(len(values_list1), len(values_list2))
    values_list1 = values_list1[:min_len]
    values_list2 = values_list2[:min_len]
    
    # 检测第一个序列的变点
    signal1 = np.asarray(values_list1, dtype=float).reshape(-1, 1)
    if model == "kernel":
        algo1 = rpt.KernelCPD(kernel="linear").fit(signal1)
        bkps1 = algo1.predict(n_bkps=n_bkps)
    else:
        algo1 = rpt.Binseg(model=model).fit(signal1)
        bkps1 = algo1.predict(epsilon=0.001)
    
    # 检测第二个序列的变点
    signal2 = np.asarray(values_list2, dtype=float).reshape(-1, 1)
    if model == "kernel":
        algo2 = rpt.KernelCPD(kernel="linear").fit(signal2)
        bkps2 = algo2.predict(n_bkps=n_bkps)
    else:
        algo2 = rpt.Binseg(model=model).fit(signal2)
        bkps2 = algo2.predict(epsilon=0.001)
    
    # 创建单个图
    plt.figure(figsize=(12, 6))
    
    # 绘制第一个序列的原始数据（蓝线）
    plt.plot(values_list1, color='blue', linewidth=2, label=f'{labels[0]} (Raw data)')
    
    # 绘制第一个序列的分段常数拟合线（蓝色虚线）
    segment_means1 = []
    start = 0
    for cp in bkps1:
        end = cp
        segment_mean = float(np.mean(signal1[start:end])) if end > start else float(signal1[start])
        segment_means1.append(segment_mean)
        if end == len(values_list1):
            end = len(values_list1) - 1
        plt.hlines(segment_mean, start, max(end, start), colors='blue', linewidth=2, linestyle='--')
        start = end
    
    # 第一个序列的竖向连接线（蓝色虚线）
    for idx in range(len(bkps1) - 1):
        x = bkps1[idx]
        y1 = segment_means1[idx]
        y2 = segment_means1[idx + 1]
        plt.vlines(x, min(y1, y2), max(y1, y2), colors='blue', linewidth=2, linestyle='--')
    
    # 绘制第二个序列的原始数据（红线）
    plt.plot(values_list2, color='red', linewidth=2, label=f'{labels[1]} (Raw data)')
    
    # 绘制第二个序列的分段常数拟合线（红色虚线）
    segment_means2 = []
    start = 0
    for cp in bkps2:
        end = cp
        segment_mean = float(np.mean(signal2[start:end])) if end > start else float(signal2[start])
        segment_means2.append(segment_mean)
        if end == len(values_list2):
            end = len(values_list2) - 1
        plt.hlines(segment_mean, start, max(end, start), colors='red', linewidth=2, linestyle='--')
        start = end
    
    # 第二个序列的竖向连接线（红色虚线）
    for idx in range(len(bkps2) - 1):
        x = bkps2[idx]
        y1 = segment_means2[idx]
        y2 = segment_means2[idx + 1]
        plt.vlines(x, min(y1, y2), max(y1, y2), colors='red', linewidth=2, linestyle='--')
    
    # 添加图例
    plt.plot([], [], color='blue', linewidth=2, linestyle='--', label=f'{labels[0]} (CUSUM process)')
    plt.plot([], [], color='red', linewidth=2, linestyle='--', label=f'{labels[1]} (CUSUM process)')
    
    # 设置标签和标题
    plt.xlabel('Timestamp', fontsize=16)
    plt.ylabel('Avg confidence score', fontsize=16)
  #  plt.title('Dual Time Series Change Point Detection', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=16)
    
    # 设置 x 轴刻度
    x_ticks = range(1, min_len + 1)
    plt.xticks(range(min_len), x_ticks)
    
    plt.tight_layout()
    plt.show()
    
    return bkps1, bkps2

if __name__ == '__main__':
    args = parameter_parser()
   # args['dataset_name'] ='cifar100'
    args['dataset_name'] ='sst5'

    args['proportion_of_group_unlearn'] =64.0
   # args['net_name'] ='resnet18'
    args['net_name'] ='pythia70m'

    args['num_epochs'] =30
    args['U_method'] ='retrain'

    args['trials'] =3
    #args['U_method'] = 'continuous_unlearn_retrain'
    #args['U_method'] = 'continuous_unlearn_NPO'
    args['U_method'] = 'continuous_unlearn_GA'

    K=50
    y_retain, y_forget, y_unseen = baseline_prep(args,K)
    #
    detect_and_plot_changepoints(y_forget)
   # detect_and_plot_dual_changepoints(y_forget,y_unseen)
    exit()


    # 绘制折线图
    plt.figure(figsize=(12, 4))
    plt.plot( y_retain, marker='o', label='Retain set')
    plt.plot( y_forget, marker='o', label='Target set')
    plt.plot( y_unseen, marker='o', label='Unseen set')

    x_ticks = range(1, len(y_forget) + 1)
    plt.xticks(ticks=range(len(y_forget)), labels=x_ticks)

    plt.xlabel('Timestamp',fontsize=18)
    plt.ylabel('Avg confidence score',fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=18, loc='upper right')

    plt.tight_layout()
    plt.show()



