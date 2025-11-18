import os

import numpy as np
import requests
import tarfile

from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from model.ResNet import resnet18_dp





def test2():
    y_target = np.array([0, 1, 0, 1, 2, 2])
    y_pred = np.array([1, 0, 0, 1, 2, 2])
    acc_target = metrics.accuracy_score(y_target, y_pred)
    ACC_list=[]
    # 遍历每个类别，计算准确率并打印
    for cls in np.unique(y_target):
        acc = np.mean(y_pred[y_target == cls] == cls)
        print(f"Class {cls} Accuracy: {acc:.2f}")
        ACC_list.append(acc)
    print("Accuracy: ", acc_target,np.mean(ACC_list))

def test3():
    # 假设有两组概率数据
    prob_1 = np.random.rand(1000)  # 1000 个随机概率
    prob_2 = np.random.rand(1000)

    # 计算 sum 和 difference
    sum_prob = prob_1 + prob_2
    diff_prob = prob_1 - prob_2

    # 画直方图
    plt.figure(figsize=(8, 6))

    # 绘制 sum 的直方图
    plt.hist(sum_prob, bins=20, density=True, alpha=0.6, color='b', label="Sum of probabilities")

    # 绘制 difference 的直方图
    plt.hist(diff_prob, bins=20, density=True, alpha=0.6, color='r', label="Difference of probabilities")

    plt.xlabel("Range of sum and difference of two probabilities", fontsize=14)
    plt.ylabel("Percentage of samples", fontsize=14)
    plt.legend()
    plt.show()

def adjust_ratio_samples(list_a, list_b, list_c, target_ratio=(2, 1, 2)):

    arr_a = np.array(list_a)
    arr_b = np.array(list_b)
    arr_c = np.array(list_c)

    # 验证输入长度一致
    assert len(arr_a) == len(arr_b)==len(arr_c)

    # 获取各类别的索引
    unseen_idx = np.where(arr_a == 0)[0]
    forget_idx = np.where(arr_a == 1)[0]
    retain_idx = np.where(arr_a == 2)[0]

    # 计算目标数量（按比例分配）
    ratio_unseen,ratio_forget,ratio_retain  = target_ratio

    min_count = min(
        len(forget_idx) // ratio_forget,
        len(unseen_idx) // ratio_unseen,
        len(retain_idx) // ratio_retain
    )

    target_forget = min_count * ratio_forget
    target_unseen = min_count * ratio_unseen
    target_retain = min_count * ratio_retain

    np.random.seed(0)

    selected_forget = np.random.choice(forget_idx, target_forget, replace=False)
    selected_unseen = np.random.choice(unseen_idx, target_unseen, replace=False)
    selected_retain = np.random.choice(retain_idx, target_retain, replace=False)


    selected_idx = np.concatenate([selected_forget, selected_unseen, selected_retain])
    selected_idx.sort()

    new_list_a = arr_a[selected_idx]
    new_list_b = arr_b[selected_idx]
    new_list_c = arr_c[selected_idx]

    return new_list_a.tolist(), new_list_b, new_list_c


def test4():
    net_name='pythia70m'
    dataset='sst5'
    save_path = os.getcwd() + f"/save/continuous_finetune/{net_name}/{dataset}/0.01/target/0/timestamp_0"
    save_path = os.getcwd() + f"/save/continuous_finetune/{net_name}/{dataset}/0.01/target/0"

    data = np.load(f"{save_path}/sample_status_converted.npy")
    print(data[1012])

if __name__ == "__main__":
    test4()