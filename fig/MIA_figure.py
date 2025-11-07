import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def MIA_SMIA_CIFAR100():
    data = {
        'Dataset': ['CIFAR-100'] * 29,
        'Model': (
            ['Pytorch-quan']*5 +
            ['AdaRound']*4 +
            ['QDrop']*5 +
            ['PACT']*5 +
            ['HAWQ']*5 +
            ['BSQ']*5
        ),
        'Bit': (
            [10, 8, 6, 4, 2] +      # Pytorch-quan
            [10, 8, 6, 4] +         # Adaround
            [10, 8, 6, 4, 2] +      # Qdrop
            [10, 8, 6, 4, 2] +      # PACT
            [10, 8, 6, 4, 2] +      # HAWQ
            [10, 8, 6, 4, 2]        # BSQ
        ),
        'Value': [
            # Pytorch-quan
            92.16, 92.09, 91.98, 87.75, 50.55,
            # Adaround
            92.64, 92.42, 92.10, 87.04,
            # Qdrop
            92.21, 92.34, 91.89, 87.02, 55.54,
            # PACT
            91.89, 91.56, 91.42, 91.20, 56.50,
            # HAWQ
            92.68, 92.82, 92.74, 92.12, 49.48,
            # BSQ
            81.31, 81.28, 80.64, 74.19, 56.03
        ]
    }
    baseline = 91.93
    bit_order = ['10', '8', '6', '4', '2']
    df = pd.DataFrame(data)
    df['Bit'] = df['Bit'].astype(str)
    df['Bit'] = pd.Categorical(df['Bit'], categories=bit_order, ordered=True)
    plt.figure(figsize=(10, 7))
    sns.lineplot(
        data=df,
        x='Bit',
        y='Value',
        hue='Model',
        marker='o'
    )
    plt.axhline(
        y=baseline,
        color='gray',
        linestyle='--',
        linewidth=1.5,
        label='Original (32-bit)'
    )
    plt.xlabel('Bit', fontsize=22)
    plt.ylabel('AUC (%)', fontsize=22)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(bottom=50)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles=handles, labels=labels, loc='lower left', fontsize=24, title_fontsize=15)
    plt.tight_layout()
    plt.show()

def MIA_MLleak_CIFAR100():
    data = {
        'Dataset': ['CIFAR-100'] * 29,
        'Model': (
            ['Pytorch-quan']*5 +
            ['AdaRound']*4 +
            ['QDrop']*5 +
            ['PACT']*5 +
            ['HAWQ']*5 +
            ['BSQ']*5
        ),
        'Bit': (
            [10, 8, 6, 4, 2] +      # Pytorch-quan
            [10, 8, 6, 4] +         # Adaround
            [10, 8, 6, 4, 2] +      # Qdrop
            [10, 8, 6, 4, 2] +      # PACT
            [10, 8, 6, 4, 2] +      # HAWQ
            [10, 8, 6, 4, 2]        # BSQ
        ),
        'Value': [
            # Pytorch-quan
            86.53, 86.42, 85.72, 73.30, 48.93,
            # Adaround
            87.51, 87.40, 85.46, 73.20,
            # Qdrop
            85.73, 86.02, 84.54, 70.63, 55.94,
            # PACT
            87.35, 86.43, 85.36, 84.94, 50.30,
            # HAWQ
            81.51, 81.21, 81.32, 81.23, 53.69,
            # BSQ
            60.76, 60.59, 60.68, 54.26, 44.56
        ]
    }
    baseline = 85.94
    bit_order = ['10', '8', '6', '4', '2']
    df = pd.DataFrame(data)
    df['Bit'] = df['Bit'].astype(str)
    df['Bit'] = pd.Categorical(df['Bit'], categories=bit_order, ordered=True)
    plt.figure(figsize=(10, 7))
    sns.lineplot(
        data=df,
        x='Bit',
        y='Value',
        hue='Model',
        marker='o'
    )
    plt.axhline(
        y=baseline,
        color='gray',
        linestyle='--',
        linewidth=1.5,
        label='Original (32-bit)'
    )
    plt.xlabel('Bit', fontsize=22)
    plt.ylabel('AUC (%)', fontsize=22)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(bottom=40)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles=handles, labels=labels, loc='lower left', fontsize=24, title_fontsize=15)
    plt.tight_layout()
    plt.show()

def MIA_RMIA_CIFAR100():
    data = {
        'Dataset': ['CIFAR-100'] * 29,
        'Model': (
            ['Pytorch-quan']*5 +
            ['AdaRound']*4 +
            ['QDrop']*5 +
            ['PACT']*5 +
            ['HAWQ']*5 +
            ['BSQ']*5
        ),
        'Bit': (
            [10, 8, 6, 4, 2] +      # Pytorch-quan
            [10, 8, 6, 4] +         # Adaround
            [10, 8, 6, 4, 2] +      # Qdrop
            [10, 8, 6, 4, 2] +      # PACT
            [10, 8, 6, 4, 2] +      # HAWQ
            [10, 8, 6, 4, 2]        # BSQ
        ),
        'Value': [
            # Pytorch-quan
            92.16, 93.37, 91.98, 91.2, 50.4,
            # Adaround
            92.64, 92.50, 92.10, 90.32,
            # Qdrop
            92.21, 93.45, 91.89, 89.76, 47.19,
            # PACT
            91.20, 92.08, 91.56, 91.78, 50.83,
            # HAWQ
            92.68, 93.98, 92.74, 92.24, 50.38,
            # BSQ
            81.31, 84.85, 80.64, 76.85, 56.96
        ]
    }
    baseline = 92.11
    bit_order = ['10', '8', '6', '4', '2']
    df = pd.DataFrame(data)
    df['Bit'] = df['Bit'].astype(str)
    df['Bit'] = pd.Categorical(df['Bit'], categories=bit_order, ordered=True)
    plt.figure(figsize=(10, 7))
    sns.lineplot(
        data=df,
        x='Bit',
        y='Value',
        hue='Model',
        marker='o'
    )
    plt.axhline(
        y=baseline,
        color='gray',
        linestyle='--',
        linewidth=1.5,
        label='Original (32-bit)'
    )
    plt.xlabel('Bit', fontsize=22)
    plt.ylabel('AUC (%)', fontsize=22)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(bottom=40)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles=handles, labels=labels, loc='lower left', fontsize=24, title_fontsize=15)
    plt.tight_layout()
    plt.show()

def MIA_SMIA_CINIC10():
    data = {
        'Dataset': ['CINIC-10'] * 29,
        'Model': (
            ['Pytorch-quan']*5 +
            ['AdaRound']*4 +
            ['QDrop']*5 +
            ['PACT']*5 +
            ['HAWQ']*5 +
            ['BSQ']*5
        ),
        'Bit': (
            [10, 8, 6, 4, 2] +      # Pytorch-quan
            [10, 8, 6, 4] +         # Adaround
            [10, 8, 6, 4, 2] +      # Qdrop
            [10, 8, 6, 4, 2] +      # PACT
            [10, 8, 6, 4, 2] +      # HAWQ
            [10, 8, 6, 4, 2]        # BSQ
        ),
        'Value': [
            # Pytorch-quan
            77.78, 77.62, 76.06, 68.72, 50.03,
            # Adaround
            76.97, 76.91, 77.30, 69.32,
            # Qdrop
            82.14, 81.56, 81.46, 75.38, 50.92,
            # PACT
            77.54, 77.41, 77.64, 76.78, 50.71,
            # HAWQ
            83.50, 83.23, 81.82, 80.92, 49.82,
            # BSQ
            70.36, 69.02, 69.50, 65.36, 53.2
        ]
    }
    baseline = 78.57
    bit_order = ['10', '8', '6', '4', '2']
    df = pd.DataFrame(data)
    df['Bit'] = df['Bit'].astype(str)
    df['Bit'] = pd.Categorical(df['Bit'], categories=bit_order, ordered=True)
    plt.figure(figsize=(10, 7))
    sns.lineplot(
        data=df,
        x='Bit',
        y='Value',
        hue='Model',
        marker='o'
    )
    plt.axhline(
        y=baseline,
        color='gray',
        linestyle='--',
        linewidth=1.5,
        label='Original (32-bit)'
    )
    plt.xlabel('Bit', fontsize=22)
    plt.ylabel('AUC (%)', fontsize=22)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(bottom=40)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles=handles, labels=labels, loc='lower left', fontsize=24, title_fontsize=15)
    plt.tight_layout()
    plt.show()

def MIA_MLleak_CINIC10():
    data = {
        'Dataset': ['CINIC-10'] * 29,
        'Model': (
            ['Pytorch-quan']*5 +
            ['AdaRound']*4 +
            ['QDrop']*5 +
            ['PACT']*5 +
            ['HAWQ']*5 +
            ['BSQ']*5
        ),
        'Bit': (
            [10, 8, 6, 4, 2] +      # Pytorch-quan
            [10, 8, 6, 4] +         # Adaround
            [10, 8, 6, 4, 2] +      # Qdrop
            [10, 8, 6, 4, 2] +      # PACT
            [10, 8, 6, 4, 2] +      # HAWQ
            [10, 8, 6, 4, 2]        # BSQ
        ),
        'Value': [
            # Pytorch-quan
            70.89, 70.85, 68.38, 58.05, 49.74,
            # Adaround
            69.38, 69.29, 69.18, 58.07,
            # Qdrop
            74.38, 74.33, 73.68, 63.42, 50.31,
            # PACT
            70.12, 70.06, 69.95, 68.86, 50.07,
            # HAWQ
            76.77, 76.74, 74.39, 69.37, 49.96,
            # BSQ
            59.59, 58.19, 58.06, 55.38, 50.15
        ]
    }
    baseline = 71.55
    bit_order = ['10', '8', '6', '4', '2']
    df = pd.DataFrame(data)
    df['Bit'] = df['Bit'].astype(str)
    df['Bit'] = pd.Categorical(df['Bit'], categories=bit_order, ordered=True)
    plt.figure(figsize=(10, 7))
    sns.lineplot(
        data=df,
        x='Bit',
        y='Value',
        hue='Model',
        marker='o'
    )
    plt.axhline(
        y=baseline,
        color='gray',
        linestyle='--',
        linewidth=1.5,
        label='Original (32-bit)'
    )
    plt.xlabel('Bit', fontsize=22)
    plt.ylabel('AUC (%)', fontsize=22)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(bottom=40)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles=handles, labels=labels, loc='lower left', fontsize=24, title_fontsize=15)
    plt.tight_layout()
    plt.show()

def MIA_RMIA_CINIC10():
    data = {
        'Dataset': ['CINIC-10'] * 29,
        'Model': (
            ['Pytorch-quan']*5 +
            ['AdaRound']*4 +
            ['QDrop']*5 +
            ['PACT']*5 +
            ['HAWQ']*5 +
            ['BSQ']*5
        ),
        'Bit': (
            [10, 8, 6, 4, 2] +      # Pytorch-quan
            [10, 8, 6, 4] +         # Adaround
            [10, 8, 6, 4, 2] +      # Qdrop
            [10, 8, 6, 4, 2] +      # PACT
            [10, 8, 6, 4, 2] +      # HAWQ
            [10, 8, 6, 4, 2]        # BSQ
        ),
        'Value': [
            # Pytorch-quan
            82.99, 84.47, 76.06, 73.85, 50.33,
            # Adaround
            76.97, 76.91, 77.30, 72.09,
            # Qdrop
            82.14, 84.13, 81.46, 78.34, 51.01,
            # PACT
            77.54, 85.25, 77.64, 82.19, 50.92,
            # HAWQ
            83.50, 86.11, 81.82, 82.93, 50.05,
            # BSQ
            70.36, 74.09, 69.50, 69.05, 54.11
        ]
    }
    baseline = 82.99
    bit_order = ['10', '8', '6', '4', '2']
    df = pd.DataFrame(data)
    df['Bit'] = df['Bit'].astype(str)
    df['Bit'] = pd.Categorical(df['Bit'], categories=bit_order, ordered=True)
    plt.figure(figsize=(10, 7))
    sns.lineplot(
        data=df,
        x='Bit',
        y='Value',
        hue='Model',
        marker='o'
    )
    plt.axhline(
        y=baseline,
        color='gray',
        linestyle='--',
        linewidth=1.5,
        label='Original (32-bit)'
    )
    plt.xlabel('Bit', fontsize=22)
    plt.ylabel('AUC (%)', fontsize=22)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(bottom=40)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles=handles, labels=labels, loc='lower left', fontsize=22, title_fontsize=15)
    plt.tight_layout()
    plt.show()

def get_mia_data():
    data_dict = {
        'MIA_SMIA_CIFAR100': {
            'data': {
                'Dataset': ['CIFAR-100'] * 29,
                'Model': (
                    ['Pytorch-quan']*5 +
                    ['AdaRound']*4 +
                    ['QDrop']*5 +
                    ['PACT']*5 +
                    ['HAWQ']*5 +
                    ['BSQ']*5
                ),
                'Bit': (
                    [10, 8, 6, 4, 2] + [10, 8, 6, 4] + [10, 8, 6, 4, 2] +
                    [10, 8, 6, 4, 2] + [10, 8, 6, 4, 2] + [10, 8, 6, 4, 2]
                ),
                'Value': [
                    92.16, 92.09, 91.98, 87.75, 50.55,
                    92.64, 92.42, 92.10, 87.04,
                    92.21, 92.34, 91.89, 87.02, 55.54,
                    91.89, 91.56, 91.42, 91.20, 56.50,
                    92.68, 92.82, 92.74, 92.12, 49.48,
                    81.31, 81.28, 80.64, 74.19, 56.03
                ]
            },
            'baseline': 91.93,
            'title': 'SMIA (CIFAR-100)'
        },
        'MIA_MLleak_CIFAR100': {
            'data': {
                'Dataset': ['CIFAR-100'] * 29,
                'Model': (
                    ['Pytorch-quan']*5 +
                    ['AdaRound']*4 +
                    ['QDrop']*5 +
                    ['PACT']*5 +
                    ['HAWQ']*5 +
                    ['BSQ']*5
                ),
                'Bit': (
                    [10, 8, 6, 4, 2] + [10, 8, 6, 4] + [10, 8, 6, 4, 2] +
                    [10, 8, 6, 4, 2] + [10, 8, 6, 4, 2] + [10, 8, 6, 4, 2]
                ),
                'Value': [
                    86.53, 86.42, 85.72, 73.30, 48.93,
                    87.51, 87.40, 85.46, 73.20,
                    85.73, 86.02, 84.54, 70.63, 55.94,
                    87.35, 86.43, 85.36, 84.94, 50.30,
                    81.51, 81.21, 81.32, 81.23, 53.69,
                    60.76, 60.59, 60.68, 54.26, 44.56
                ]
            },
            'baseline': 85.94,
            'title': 'MLleak (CIFAR-100)'
        },
        'MIA_RMIA_CIFAR100': {
            'data': {
                'Dataset': ['CIFAR-100'] * 29,
                'Model': (
                    ['Pytorch-quan']*5 +
                    ['AdaRound']*4 +
                    ['QDrop']*5 +
                    ['PACT']*5 +
                    ['HAWQ']*5 +
                    ['BSQ']*5
                ),
                'Bit': (
                    [10, 8, 6, 4, 2] + [10, 8, 6, 4] + [10, 8, 6, 4, 2] +
                    [10, 8, 6, 4, 2] + [10, 8, 6, 4, 2] + [10, 8, 6, 4, 2]
                ),
                'Value': [
                    92.16, 93.37, 91.98, 91.2, 50.4,
                    92.64, 92.50, 92.10, 90.32,
                    92.21, 93.45, 91.89, 89.76, 47.19,
                    91.20, 92.08, 91.56, 91.78, 50.83,
                    92.68, 93.98, 92.74, 92.24, 50.38,
                    81.31, 84.85, 80.64, 76.85, 56.96
                ]
            },
            'baseline': 92.11,
            'title': 'RMIA (CIFAR-100)'
        },
        'MIA_SMIA_CINIC10': {
            'data': {
                'Dataset': ['CINIC-10'] * 29,
                'Model': (
                    ['Pytorch-quan']*5 +
                    ['AdaRound']*4 +
                    ['QDrop']*5 +
                    ['PACT']*5 +
                    ['HAWQ']*5 +
                    ['BSQ']*5
                ),
                'Bit': (
                    [10, 8, 6, 4, 2] + [10, 8, 6, 4] + [10, 8, 6, 4, 2] +
                    [10, 8, 6, 4, 2] + [10, 8, 6, 4, 2] + [10, 8, 6, 4, 2]
                ),
                'Value': [
                    77.78, 77.62, 76.06, 68.72, 50.03,
                    76.97, 76.91, 77.30, 69.32,
                    82.14, 81.56, 81.46, 75.38, 50.92,
                    77.54, 77.41, 77.64, 76.78, 50.71,
                    83.50, 83.23, 81.82, 80.92, 49.82,
                    70.36, 69.02, 69.50, 65.36, 53.2
                ]
            },
            'baseline': 78.57,
            'title': 'SMIA (CINIC-10)'
        },
        'MIA_MLleak_CINIC10': {
            'data': {
                'Dataset': ['CINIC-10'] * 29,
                'Model': (
                    ['Pytorch-quan']*5 +
                    ['AdaRound']*4 +
                    ['QDrop']*5 +
                    ['PACT']*5 +
                    ['HAWQ']*5 +
                    ['BSQ']*5
                ),
                'Bit': (
                    [10, 8, 6, 4, 2] + [10, 8, 6, 4] + [10, 8, 6, 4, 2] +
                    [10, 8, 6, 4, 2] + [10, 8, 6, 4, 2] + [10, 8, 6, 4, 2]
                ),
                'Value': [
                    70.89, 70.85, 68.38, 58.05, 49.74,
                    69.38, 69.29, 69.18, 58.07,
                    74.38, 74.33, 73.68, 63.42, 50.31,
                    70.12, 70.06, 69.95, 68.86, 50.07,
                    76.77, 76.74, 74.39, 69.37, 49.96,
                    59.59, 58.19, 58.06, 55.38, 50.15
                ]
            },
            'baseline': 71.55,
            'title': 'MLleak (CINIC-10)'
        },
        'MIA_RMIA_CINIC10': {
            'data': {
                'Dataset': ['CINIC-10'] * 29,
                'Model': (
                    ['Pytorch-quan']*5 +
                    ['AdaRound']*4 +
                    ['QDrop']*5 +
                    ['PACT']*5 +
                    ['HAWQ']*5 +
                    ['BSQ']*5
                ),
                'Bit': (
                    [10, 8, 6, 4, 2] + [10, 8, 6, 4] + [10, 8, 6, 4, 2] +
                    [10, 8, 6, 4, 2] + [10, 8, 6, 4, 2] + [10, 8, 6, 4, 2]
                ),
                'Value': [
                    82.99, 84.47, 76.06, 73.85, 50.33,
                    76.97, 76.91, 77.30, 72.09,
                    82.14, 84.13, 81.46, 78.34, 51.01,
                    77.54, 85.25, 77.64, 82.19, 50.92,
                    83.50, 86.11, 81.82, 82.93, 50.05,
                    70.36, 74.09, 69.50, 69.05, 54.11
                ]
            },
            'baseline': 82.99,
            'title': 'RMIA (CINIC-10)'
        }
    }
    return data_dict

def plot_mia_6in1():
    data_dict = get_mia_data()
    print("xxxx")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    fig, axes = plt.subplots(2, 3, figsize=(22, 12), sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.15, hspace=0.22)
    bit_order = ['10', '8', '6', '4', '2']

    keys = [
        'MIA_SMIA_CIFAR100', 'MIA_MLleak_CIFAR100', 'MIA_RMIA_CIFAR100',
        'MIA_SMIA_CINIC10', 'MIA_MLleak_CINIC10', 'MIA_RMIA_CINIC10'
    ]
    titles = [
        'SMIA', 'MLleak', 'RMIA',
        'SMIA', 'MLleak', 'RMIA'
    ]
    datasets = ['CIFAR-100', 'CIFAR-100', 'CIFAR-100', 'CINIC-10', 'CINIC-10', 'CINIC-10']

    for idx, key in enumerate(keys):
        row, col = divmod(idx, 3)
        ax = axes[row, col]
        entry = data_dict[key]
        df = pd.DataFrame(entry['data'])
        df['Bit'] = df['Bit'].astype(str)
        df['Bit'] = pd.Categorical(df['Bit'], categories=bit_order, ordered=True)
        sns.lineplot(
            data=df,
            x='Bit',
            y='Value',
            hue='Model',
            marker='o',
            ax=ax,
            estimator=None
        )
        ax.axhline(
            y=entry['baseline'],
            color='gray',
            linestyle='--',
            linewidth=1.5,
            label='Original (32-bit)'
        )
        ax.set_title(f"{titles[col]} ({datasets[idx]})", fontsize=18)
        ax.set_xlabel('Bit', fontsize=16)
        ax.set_ylabel('AUC (%)', fontsize=16)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_ylim(bottom=40)
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
        if row == 0 and col == 0:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=handles, labels=labels, loc='lower left', fontsize=14, title_fontsize=13)
        else:
            ax.get_legend().remove()

    fig.suptitle('MIA Results on CIFAR-100 and CINIC-10', fontsize=26, y=1.03)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

if __name__ == "__main__":
    # MIA_MLleak_CIFAR100()
    # MIA_SMIA_CIFAR100()
    # MIA_MLleak_CINIC10()
    # MIA_SMIA_CINIC10()
    # plot_mia_6in1()
    MIA_SMIA_CIFAR100()
  #  MIA_RMIA_CINIC10()
