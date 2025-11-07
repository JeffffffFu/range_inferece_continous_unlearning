import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# =====================================
# 1. 生成多条合成时间序列
# =====================================
def generate_sequence(length=100, first_cp=30, noise=0.02):
    """
    生成时间序列：先上升突变，然后过30个时间点后下降突变
    first_cp: 第一个突变点位置（上升）
    第二个突变点位置：first_cp + 30（下降）
    值域控制在0-1之间
    """
    data, labels = [], []
    value = 0.5  # 初始值设为0.5，在0-1中间
    second_cp = first_cp + 30  # 第二个突变点在第一个之后30个时间点
    
    for i in range(length):
        if i == first_cp:
            # 第一个突变点：上升
            change_magnitude = np.random.uniform(0.1, 0.3)  # 减小变化幅度
            value = min(1.0, value + change_magnitude)  # 上升，但不超过1
            labels.append(1)  # 上升突变
        elif i == second_cp:
            # 第二个突变点：下降
            change_magnitude = np.random.uniform(0.1, 0.3)  # 减小变化幅度
            value = max(0.0, value - change_magnitude)  # 下降，但不低于0
            labels.append(2)  # 下降突变
        else:
            labels.append(0)  # 非突变点
        
        val = value + np.random.normal(0, noise)
        val = np.clip(val, 0.0, 1.0)  # 确保值在0-1范围内
        data.append(val)
    return np.array(data), np.array(labels)


def generate_dataset(n_series=5, length=100):
    """
    生成数据集：每条序列都有固定的模式（先上升后下降，间隔30个时间点）
    """
    all_data, all_labels = [], []
    for _ in range(n_series):
        # 随机选择第一个突变点位置，确保第二个突变点不超出序列长度
        first_cp = np.random.choice(range(20, length - 50))  # 确保第二个突变点在序列内
        data, labels = generate_sequence(length, first_cp)
        all_data.append(data)
        all_labels.append(labels)
    return all_data, all_labels


# 增加训练集数量到10条序列
train_data, train_labels = generate_dataset(n_series=10, length=100)
test_data, test_labels = generate_dataset(n_series=3, length=100)


# =====================================
# 2. 构造滑动窗口数据
# =====================================
def create_windows_from_series(series_list, label_list, window_size=10):
    X, y = [], []
    for data, labels in zip(series_list, label_list):
        for i in range(len(data) - window_size):
            X.append(data[i:i + window_size])
            # 取窗口中间位置的时间点标签 (t+5)
            mid_idx = i + window_size // 2  # i + 5
            y.append(labels[mid_idx])
    X = np.array(X).reshape(len(X), window_size, 1)
    y = np.array(y)
    return X, y


window_size = 2
X_train, y_train = create_windows_from_series(train_data, train_labels, window_size)
X_test, y_test = create_windows_from_series(test_data, test_labels, window_size)

X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)


# =====================================
# 3. 定义双向 LSTM 模型（三分类）
# =====================================
class BiLSTMChangePoint(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size,
                            batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 3)  # 改为3分类：0,1,2

    def forward(self, x):
        # x shape: (batch_size, seq_len, 1) = (batch_size, 10, 1)
        out, _ = self.lstm(x)  # out shape: (batch_size, 10, hidden_size*2)
        
        # 取中间位置t+5的输出 (索引5，因为窗口是t到t+9，中间是t+5)
        seq_len = x.size(1)
        mid_idx = seq_len // 2  # 10 // 2 = 5
        mid_out = out[:, mid_idx, :]  # 取中间时间步的输出
        
        return self.fc(mid_out)


model = BiLSTMChangePoint()

# 分析训练数据的类别分布，计算平衡权重
print("=== 训练数据类别分布分析 ===")
all_train_labels = np.concatenate(train_labels)
class_counts = np.bincount(all_train_labels)
print(f"类别0（非突变点）数量: {class_counts[0]}")
print(f"类别1（上升突变）数量: {class_counts[1]}")
print(f"类别2（下降突变）数量: {class_counts[2]}")

# 计算平衡权重（类别数量越多，权重越小）
total_samples = len(all_train_labels)
weights = total_samples / (3 * class_counts)
print(f"计算的权重: {weights}")

criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# =====================================
# 4. 训练模型
# =====================================
for epoch in range(600):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 2 == 0:
        print(f"Epoch {epoch:02d}, Loss = {loss.item():.4f}")

# =====================================
# 5. 测试与逐点输出预测（三分类）
# =====================================
model.eval()
with torch.no_grad():
    preds = model(X_test)
    preds_cls = torch.argmax(preds, dim=1).numpy()

# 三分类评估
acc = accuracy_score(y_test, preds_cls)
prec = precision_score(y_test, preds_cls, average='macro')  # 使用macro平均
rec = recall_score(y_test, preds_cls, average='macro')
f1 = f1_score(y_test, preds_cls, average='macro')

print("\n=== Evaluation on Test Set (3-class) ===")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-score:  {f1:.4f}")

# 详细类别分析
print("\n=== Detailed Class Analysis ===")
test_class_counts = np.bincount(y_test)
pred_class_counts = np.bincount(preds_cls)
print("测试集真实类别分布:", test_class_counts)
print("预测类别分布:", pred_class_counts)


# =====================================
# 6. 可视化训练集序列和突变点（三分类）
# =====================================
print("\n=== Training Set Visualization (3-class) ===")
plt.figure(figsize=(12, 16))  # 增加高度以容纳10个序列
for idx, (data, true_lbl) in enumerate(zip(train_data, train_labels)):
    plt.subplot(len(train_data), 1, idx + 1)
    plt.plot(data, label=f"Raw data", color='blue')
    plt.title(f"Training Series {idx + 1}")

    # 标记真实突变点（三分类）
    up_cp_labeled = False
    down_cp_labeled = False
    for i, v in enumerate(true_lbl):
        if v == 1:  # 上升突变
            plt.axvline(i, color='red', linestyle='--', alpha=0.8,
                        label='Entry' if not up_cp_labeled else "")
            up_cp_labeled = True
        elif v == 2:  # 下降突变
            plt.axvline(i, color='orange', linestyle=':', alpha=0.8,
                        label='Exist' if not down_cp_labeled else "")
            down_cp_labeled = True
    
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

plt.show()


# =====================================
# 7. 可视化每条测试序列的预测与真实突变点
# =====================================
def reconstruct_predictions(test_series, test_labels, window_size, model):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for seq in test_series:
            # 初始化预测序列，全为0
            preds = [0] * len(seq)
            
            # 对每个窗口进行预测
            for i in range(len(seq) - window_size):
                x = torch.FloatTensor(seq[i:i + window_size]).view(1, window_size, 1)
                y_pred = torch.argmax(model(x), dim=1).item()
                
                # 将预测结果放到窗口中间位置 (i + window_size//2)
                mid_idx = i + window_size // 2
                preds[mid_idx] = y_pred
            
            all_preds.append(preds)
    return all_preds


pred_series = reconstruct_predictions(test_data, test_labels, window_size, model)
print(f'pred:{pred_series}')
# 绘图（三分类）- 只显示第一个测试序列
plt.figure(figsize=(15, 6))
data, true_lbl, pred_lbl = test_data[0], test_labels[0], pred_series[0]
plt.plot(data, label=f"Raw data", color='blue')
#plt.title(f"Test Series 1")

# 标记预测突变点（三分类）- 使用虚线和大点标记
for i, v in enumerate(pred_lbl):
    if v == 1:  # 预测上升突变
        plt.axvline(i, color='red', linestyle='--', linewidth=3, alpha=0.8)
    elif v == 2:  # 预测下降突变
        plt.axvline(i, color='magenta', linestyle=':', linewidth=3, alpha=0.8)
plt.tick_params(axis='both', which='major', labelsize=18)

plt.xlabel('Timestamp', fontsize=22)
plt.ylabel('Confidence score', fontsize=22)
plt.legend(loc='upper left', fontsize=22)
plt.tight_layout()

plt.show()
