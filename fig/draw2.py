from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
def transfer_model_acc():
    # 创建新的数据（按给定的表格）
    data = np.array([
        [79.25, 90.89, 86.85],  # DenseNet (第1行)
        [79.80, 91.84, 85.91],  # SimpleCNN (第2行)
        [79.98, 87.80, 83.40]  # ResNet18 (第3行)
    ])

    # 设定行列名称，保持正确对应
    models = ["ResNet18", "SimpleCNN", "DenseNet"]
    df = pd.DataFrame(data, index=["DenseNet", "SimpleCNN", "ResNet18"], columns=models)

    # 格式化数据（显示百分比）
    formatted_data = df.map(lambda x: f'{x:.2f}%')

    # 绘制热图
    plt.figure(figsize=(6, 5))
    sns.heatmap(df, annot=formatted_data, cmap="coolwarm", linewidths=0.5, fmt="",
                annot_kws={"fontsize": 14})

    # 调整坐标轴字体大小
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # 显示图像
    plt.show()


def transfer_model_per_class():
    data_types = {
        "Unseen": np.array([
            [79.06, 90.88, 87.93],  # 第一行：DenseNet, SimpleCNN, DenseNet
            [78.65, 92.44, 87.79],  # 第二行
            [79.10, 85.58, 81.09]  # 第三行：ResNet18放在第一列
        ]),
        "Forget": np.array([
            [79.16, 91.5, 86.45],
            [79.42, 92.21, 86.27],
            [79.7, 89.05, 84.22]
        ]),
        "Retain": np.array([
            [79.83, 90.36, 87.93],
            [79.25, 91.02, 89.79],
            [80.83, 88.42, 81.09]
        ])
    }

    models = ["ResNet18", "SimpleCNN", "DenseNet"]

    for data_name, data in data_types.items():
        df = pd.DataFrame(data, index=["DenseNet", "SimpleCNN", "ResNet18"], columns=models)
        formatted_data = df.map(lambda x: f'{x:.2f}%')

        plt.figure(figsize=(8, 6))
        heatmap = sns.heatmap(df, annot=formatted_data, cmap="coolwarm",
                              linewidths=0.5, fmt="", annot_kws={"fontsize": 18},
                              vmin=75, vmax=95)

        # 调整坐标轴字体大小
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
     #   heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45)
     #   heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0)
        plt.tight_layout()
        plt.show()


def transfer_dataset_acc():
    # 创建新的数据（按给定的表格）
    data = np.array([
        [58.17, 79.16, 92.50],  # TinyImageNet (第1行)
        [58.53, 79.98, 91.69],  # CIFAR-100 (第2行)
        [59.26,78.65,90.99]  # CINIC-10 (第3行)
    ])

    # 设定行列名称，保证数据正确对应
    models = ["CINIC-10", "CIFAR-100", "TinyImageNet"]
    df = pd.DataFrame(data, index=["TinyImageNet", "CIFAR-100", "CINIC-10"], columns=models)

    # 格式化数据（显示百分比）
    formatted_data = df.map(lambda x: f'{x:.2f}%')

    # 绘制热图
    plt.figure(figsize=(6, 5))
    sns.heatmap(df, annot=formatted_data, cmap="coolwarm", linewidths=0.5, fmt="",
                annot_kws={"fontsize": 14})

    # 调整坐标轴字体大小
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # 显示图像
    plt.show()
def transfer_dataset_per_class():
    data_types = {
        "Unseen": np.array([
            [56.17, 78.2, 93.6],  # 第一行：DenseNet, SimpleCNN, DenseNet
            [55.73, 79.1, 93.46],  # 第二行
            [57.84, 79.05, 92.66]  # 第三行：ResNet18放在第一列
        ]),
        "Forget": np.array([
            [50.78, 76.46, 91.24],
            [52.62, 79.7, 91.15],
            [55.28, 76.11, 91.08]
        ]),
        "Retain": np.array([
            [63.41, 79.75, 92.69],
            [63.57, 80.83, 92.02],
            [63.84, 79.12, 91.01]
        ])
    }

    models = ["CINIC-10", "CIFAR-100", "TinyImageNet"]

    for data_name, data in data_types.items():
        df = pd.DataFrame(data, index=["TinyImageNet", "CIFAR-100", "CINIC-10"], columns=models)
        formatted_data = df.map(lambda x: f'{x:.2f}%')

        plt.figure(figsize=(8, 6))
        heatmap = sns.heatmap(df, annot=formatted_data, cmap="coolwarm",
                              linewidths=0.5, fmt="", annot_kws={"fontsize": 18},
                              vmin=75, vmax=95)

        # 调整坐标轴字体大小
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
     #   heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45)
     #   heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0)
        plt.tight_layout()
        plt.show()

def transfer_algo_acc():
    # 创建新的数据（保留原始百分比值，去掉%符号）
    data = np.array([
        [78.87, 73.11, 79.96],  # Sparsity
        [78.60, 74.89, 77.85],  # GA
        [79.98, 69.02, 72.31]  # Retrain
    ])

    # 设定行列名称
    models = ["Retrain", "GA", "Sparsity"]
    df = pd.DataFrame(data, index=["Sparsity", "GA", "Retrain"], columns=models)


    # 格式化数据，添加百分号
    formatted_data = df.map(lambda x: f'{x:.2f}%')

    # 绘制热图
    plt.figure(figsize=(6, 5))
    sns.heatmap(df, annot=formatted_data, cmap="coolwarm", linewidths=0.5, fmt="",
                annot_kws={"fontsize": 14})

    # 调整坐标轴字体大小
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # 显示图像
    plt.show()


def transfer_algo_per_class():
    data_types = {
        "Unseen": np.array([
            [78.87, 85.01, 85.41],  # 第一行：DenseNet, SimpleCNN, DenseNet
            [79.05, 86.19, 84.66],  # 第二行
            [79.1, 73.52, 73.68]  # 第三行：ResNet18放在第一列
        ]),
        "Forget": np.array([
            [76.93, 62.47, 76.67],
            [78.36, 68.70, 75.57],
            [79.70, 65.82, 72.26]
        ]),
        "Retain": np.array([
            [79.60, 71.03, 78.56],
            [80.16, 71.39, 74.56],
            [80.83, 68.84, 71.35]
        ])
    }

    models = ["Retrain", "GA", "Sparsity"]

    for data_name, data in data_types.items():
        df = pd.DataFrame(data, index=["Sparsity", "GA", "Retrain"], columns=models)
        formatted_data = df.map(lambda x: f'{x:.2f}%')

        plt.figure(figsize=(8, 6))
        heatmap = sns.heatmap(df, annot=formatted_data, cmap="coolwarm",
                              linewidths=0.5, fmt="", annot_kws={"fontsize": 18},
                              vmin=75, vmax=95)

        # 调整坐标轴字体大小
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
     #   heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45)
     #   heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0)
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    #transfer_model_acc()
   # transfer_dataset_acc()
    transfer_algo_per_class()