from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px

def group_deletion():
    import matplotlib.pyplot as plt

    import matplotlib.pyplot as plt

    # Data
    unlearning_methods = ['Scratch', 'SISA', 'Sparsity', 'SCRUB']
    percentage_unlearned_samples = [100, 200, 300, 500, 800]
    scratch_acc = [82.44, 82.12, 82.05, 81.94, 80.31]
    sisa_acc = [84.38, 84.72, 84.55, 84.11, 83.85]
    sparsity_acc = [63.89, 62.85, 62.08, 59.72, 59.13]
    scrub_acc = [58.42, 57.99, 57.77, 57.64, 57.21]

    # Plotting
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))  # Remove sharey=True

    # Subplot 1: Scratch
    axes[0].plot(percentage_unlearned_samples, scratch_acc, marker='o', linestyle='-', color='blue', label="Scratch")
    axes[0].set_xlabel(r"Number of Unlearned Samples", fontsize=16)
    axes[0].set_ylabel("Attack Accuracy (%)", fontsize=18)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend(fontsize=18)
    axes[0].set_ylim(min(scratch_acc) - 2, max(scratch_acc) + 2)  # Adaptive Y-axis range
    axes[0].set_xlim(50, 850)

    # Subplot 2: SISA
    axes[1].plot(percentage_unlearned_samples, sisa_acc, marker='o', linestyle='-', color='green', label="SISA")
    axes[1].set_xlabel(r"Number of Unlearned Samples", fontsize=16)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend(fontsize=18)
    axes[1].set_ylim(min(sisa_acc) - 2, max(sisa_acc) + 2)  # Adaptive Y-axis range

    # Subplot 3: Sparsity
    axes[2].plot(percentage_unlearned_samples, sparsity_acc, marker='o', linestyle='-', color='red', label="Sparsity")
    axes[2].set_xlabel(r"Number of Unlearned Samples", fontsize=16)
    axes[2].grid(True, linestyle='--', alpha=0.6)
    axes[2].legend(fontsize=18)
    axes[2].set_ylim(min(sparsity_acc) - 2, max(sparsity_acc) + 2)  # Adaptive Y-axis range

    # Subplot 4: SCRUB
    axes[3].plot(percentage_unlearned_samples, scrub_acc, marker='o', linestyle='-', color='purple', label="SCRUB")
    axes[3].set_xlabel(r"Number of Unlearned Samples", fontsize=16)
    axes[3].grid(True, linestyle='--', alpha=0.6)
    axes[3].legend(fontsize=18)
    axes[3].set_ylim(min(scrub_acc) - 2, max(scrub_acc) + 2)  # Adaptive Y-axis range

    # Adjust layout and show plot
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


import matplotlib.pyplot as plt

def plot_impact_of_parameters():
    # Define all plot configurations
    plot_configs = [
        {
            'title': 'Impact of Shadow Set Size',
            'xlabel': 'Percentage of the shadow training set',
            'x_values': ['6%', '18%', '30%', '48%', '60%'],
            'y_values': {
                'CIFAR-100': [79.64, 79.78, 79.98, 80.18, 80.38],
                'CINIC-10': [59.16,59.22,59.26,59.28,59.33]
            },
            'ylim': (58, 81)
        },
        {
            'title': 'Impact of Forget Set Size',
            'xlabel': 'Percentage of training set',
            'x_values': ['2%', '10%', '20%', '30%', '40%'],
            'y_values': {
                'CIFAR-100': [79.98, 80.05, 81.37, 81.53, 82.49],
                'CINIC-10': [59.26,59.34,59.57,59.78,59.97]
            },
            'ylim': (58, 83)
        },
        {
            'title': 'Impact of Size Ratios',
            'xlabel': 'unseen:forget:retain',
            'x_values': ['1:1:1', '2:1:2', '2:1:4', '4:1:2'],
            'y_values': {
                'CIFAR-100': [79.98, 79.20, 78.64, 74.96],
                'CINIC-10': [59.26,58.31,56.99,35.33]
            },
            'ylim': (32, 81)
        }
    ]

    # Plot each configuration
    for config in plot_configs:
        plt.figure(figsize=(8, 6))

        # Plot each dataset
        colors = {'CIFAR-100': 'b', 'CINIC-10': 'r'}
        markers = {'CIFAR-100': 'o', 'CINIC-10': 's'}

        for label, y_values in config['y_values'].items():
            plt.plot(config['x_values'], y_values,
                     marker=markers[label],
                     linestyle='-',
                     color=colors[label],
                     label=label)

        # Set plot properties
        plt.ylim(config['ylim'])
        plt.xlabel(config['xlabel'], fontsize=28)
        plt.ylabel('F1-Score (%)', fontsize=28)
        plt.legend(loc='best' if 'Size Ratios' in config['title'] else 'center right', fontsize=22)
        plt.grid(True)
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.subplots_adjust(top=0.95, bottom=0.16)

        plt.show()

#DP attack acc

def dp_1():
    # 数据
    sigma = [0.5, 1, 1.5,2]
    accuracy = [36.9,35.48,34.33,33.71]

    # 创建折线图
    plt.figure(figsize=(8, 6))
    plt.plot(sigma, accuracy, marker='o', linestyle='-', color='b', label='Attack Accuracy (%)')
    plt.ylim(33,38)

    # 添加标题和标签
    plt.xlabel(r'Privacy budget $\sigma$', fontsize=28)
    plt.ylabel('F1-Score (%)', fontsize=28)
    #plt.legend(loc='best', fontsize=22)
    # 设置网格
    plt.grid(True)
    plt.subplots_adjust(top=0.9, bottom=0.2)  # 增加 bottom 值以防止横坐标被遮挡

    # 设置横纵坐标字体大小
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    # 显示图表
    plt.show()
import matplotlib.ticker as ticker

#DP train_acc and test_acc
def dp_2():
    sigma = [0.5, 1, 1.5, 2]
    train_accuracy = [61.8,53.4,51.02,48.36]
    test_accuracy = [58.1,51.68,49.77,47.16]

    # 创建图形
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # 左坐标轴
    ax1.plot(sigma, test_accuracy, marker='o', linestyle='-', color='b', label='Test Accuracy (%)')
    ax1.set_xlabel(r'Noise Scale $\sigma$', fontsize=28)
    ax1.set_ylabel('Test Accuracy (%)', fontsize=24, color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_ylim(46, 62)  # 设定Y轴范围
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))

    # 右坐标轴
    ax2 = ax1.twinx()
    ax2.plot(sigma, train_accuracy, marker='s', linestyle='--', color='r', label='Train Accuracy (%)')
    ax2.set_ylabel('Train Accuracy (%)', fontsize=24, color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_ylim(46, 62)  # 设定Y轴范围
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))

    # 添加网格
    ax1.grid(True, linestyle='--', linewidth=0.5)

    # 设置横纵坐标字体大小
    ax1.tick_params(axis='both', labelsize=20)
    ax2.tick_params(axis='both', labelsize=20)
    plt.subplots_adjust(top=0.9, bottom=0.15)  # 增加 bottom 值以防止横坐标被遮挡

    # 显示图例
    # ax1.legend(loc='upper right', fontsize=18, frameon=True)
    # ax2.legend(loc='upper right', fontsize=18, frameon=True)
    # 显示图表
    plt.show()


def dp_3():
    eps = [2.0, 4.5, 8.6, 40]
    eps = ['40','8.6', '5.0', '2.0']

    test_accuracy = [58.1,51.68,49.77,47.16]
   # test_accuracy = [47.16,49.77,51.68,58.1]
 #   F1 = [33.71,34.60,36.27,36.98]
    F1 = [36.98,36.27,34.60,33.71]

    # 创建图形
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # 左坐标轴
    ax1.plot(eps, test_accuracy, marker='o', linestyle='-', color='b', label='Test Accuracy (%)')
    ax1.set_xlabel(r'Privacy Budget $\epsilon$', fontsize=20)
    ax1.set_ylabel('Test Accuracy (%)', fontsize=20, color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_ylim(46, 60)  # 设定Y轴范围
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))

    # 右坐标轴
    ax2 = ax1.twinx()
    ax2.plot(eps, F1, marker='s', linestyle='--', color='r', label='Overall F1-score (%)')
    ax2.set_ylabel('Overall F1-score (%)', fontsize=20, color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_ylim(33, 38)  # 设定Y轴范围
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))

    # 添加网格
    ax1.grid(True, linestyle='--', linewidth=0.5)

    # 设置横纵坐标字体大小
    ax1.tick_params(axis='both', labelsize=20)
    ax2.tick_params(axis='both', labelsize=20)
    plt.subplots_adjust(top=0.9, bottom=0.15)  # 增加 bottom 值以防止横坐标被遮挡

    # 显示图例
    # ax1.legend(loc='upper right', fontsize=18, frameon=True)
    # ax2.legend(loc='upper right', fontsize=18, frameon=True)
    # 显示图表
    plt.show()

def ablation_study_scratch():


    # 数据
    datasets = ['CIFAR-10', 'CIFAR-100', 'CINIC-10', 'TinyImageNet']
    methods = ['Scratch', '-TL', '-max', '-Conf', '-Var', 'All']
    data = {
        'CIFAR-10': [53.3, 59.2, 58.85, 58.51, 59.2],
        'CIFAR-100': [69.79, 82.12, 82.12, 82.29, 82.29],
        'CINIC-10': [51.56, 61.81, 61.81, 61.63, 61.81],
        'TinyImageNet': [68.23, 89.76, 89.06, 89.93, 90.11]
    }

    # 颜色（使用更美观的颜色）
    colors = ['#a6cee3', '#fdbf6f', '#b2df8a', '#fb9a99', '#cab2d6', '#ffff99']

    # 创建子图（1 行 4 列）
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # 设置全局字体大小
    plt.rcParams.update({'font.size': 18})

    # 绘制每个数据集的柱状图
    for i, dataset in enumerate(datasets):
        ax = axes[i]
        x = np.arange(len(methods[1:]))  # x 轴位置
        values = data[dataset]  # 当前数据集的值

        # 绘制柱状图
        bars = ax.bar(x, values, color=colors[:len(values)], edgecolor='black', linewidth=1.0, alpha=0.9, width=0.6)

        # 设置标题和标签
        ax.set_title(dataset, fontsize=18, fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(methods[1:], fontsize=18, rotation=45, ha='right')
        ax.set_ylabel('Attack Accuracy (%)', fontsize=18, fontweight='bold')
        ax.set_ylim(0, max(values) + 10)  # 设置 y 轴范围，留出一些空间

        # 网格线
        ax.grid(True, linestyle='--', alpha=0.6, axis='y')

        # 在柱子上方标注数值
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}',
                    ha='center', va='bottom', fontsize=16)

    # 调整布局
    plt.tight_layout()
    plt.show()

def ablation_study_sprasity():

    # 数据
    datasets = ['CIFAR-10', 'CIFAR-100', 'CINIC-10', 'TinyImageNet']
    methods = ['Scratch', '-TL', '-max', '-Conf', '-Var', 'All']
    data = {
        'CIFAR-10': [46.71, 48.09, 48.61, 48.09, 48.09],
        'CIFAR-100': [60.7, 60.24, 64.58, 60.24, 60.42],
        'CINIC-10': [41.15, 48.61, 50.35, 51.22, 50.69],
        'TinyImageNet': [57.29, 72.22, 69.79, 69.27, 69.72]
    }

    # 颜色（使用更美观的颜色）
    colors = ['#a6cee3', '#fdbf6f', '#b2df8a', '#fb9a99', '#cab2d6', '#ffff99']

    # 创建子图（1 行 4 列）
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # 设置全局字体大小
    plt.rcParams.update({'font.size': 18})

    # 绘制每个数据集的柱状图
    for i, dataset in enumerate(datasets):
        ax = axes[i]
        x = np.arange(len(methods[1:]))  # x 轴位置
        values = data[dataset]  # 当前数据集的值

        # 绘制柱状图
        bars = ax.bar(x, values, color=colors[:len(values)], edgecolor='black', linewidth=1.0, alpha=0.9, width=0.6)

        # 设置标题和标签
        ax.set_title(dataset, fontsize=18, fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(methods[1:], fontsize=18, rotation=45, ha='right')
        ax.set_ylabel('Attack Accuracy (%)', fontsize=18, fontweight='bold')
        ax.set_ylim(0, max(values) + 10)  # 设置 y 轴范围，留出一些空间

        # 网格线
        ax.grid(True, linestyle='--', alpha=0.6, axis='y')

        # 在柱子上方标注数值
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height, f'{height:.2f}',
                    ha='center', va='bottom', fontsize=16)

    # 调整布局
    plt.tight_layout()
    plt.show()






def find_optimal_features():
    datasets = {
        "Well-to-Well": np.array([
            [35.82, 21.56, 27.94, 47.07],  # CP
            [36.73, 28.42, 27.57, 45.89],  # DF
            [36.76, 32.36, 14.70, 48.12],  # SM
            [37.91, 32.59, 28.57, 46.00],  # CT
            [37.73, 34.07, 28.47, 45.09]  # CDS
        ]),
        "Well-to-Over": np.array([
            [42.80, 27.37, 37.06, 54.49],  # CP
            [40.20, 26.89, 36.82, 50.05],  # DF
            [42.64, 39.22, 15.29, 56.12],  # SM
            [45.91, 39.74, 23.59, 58.72],  # CT
            [46.13, 35.00, 35.14, 57.60]  # CDS
        ]),
        "Over-to-Well": np.array([
            [40.55, 32.19, 27.45, 51.57],  # CP
            [44.35, 42.47, 34.30, 50.63],  # DF
            [42.44, 43.82, 15.21, 52.96],  # SM
            [45.88, 50.13, 37.60, 48.06],  # CT
            [47.60, 50.59, 38.10, 51.17]  # CDS
        ]),
        "Over-to-Over": np.array([
            [45.24, 33.99, 37.09, 55.88],  # CP
            [48.35, 39.37, 40.81, 57.03],  # DF
            [46.77, 45.97, 16.84, 59.91],  # SM
            [53.38, 51.77, 44.19, 60.06],  # CT
            [52.74, 49.63, 41.83, 60.27]  # CDS
        ])
    }

    # Common labels
    row_labels = ["CP", "DF", "SM", "CT", "CDS"]
    col_labels = ["Overall", "Unseen", "Forget", "Retain"]

    # Plot each dataset
    for scenario, data in datasets.items():
        df = pd.DataFrame(data, index=row_labels, columns=col_labels)

        plt.figure(figsize=(7, 6))
        annot_data = np.array([["{:.2f}%".format(val) for val in row] for row in data])

        sns.heatmap(df, annot=annot_data, fmt="", cmap="YlGnBu",
                    linewidths=0.5, cbar=False, annot_kws={"size": 16})

      #  plt.title(f"Performance Metrics: {scenario} Scenario", fontsize=18, pad=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlabel("F1-score", fontsize=18)
        plt.ylabel("Feature Construction Method", fontsize=18)
        plt.xticks(rotation=0)

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    plot_impact_of_parameters()