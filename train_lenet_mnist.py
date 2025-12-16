"""
使用LeNet网络训练MNIST数据集的脚本
打印训练过程中的训练准确率
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os


class LeNet(nn.Module):
    """LeNet-5网络结构"""
    
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        # 第一个卷积层：1个输入通道，6个输出通道，5x5卷积核
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        # 平均池化层：2x2
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        # 第二个卷积层：6个输入通道，16个输出通道，5x5卷积核
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        # 平均池化层：2x2
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # 全连接层1：16*5*5 = 400 -> 120
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # 全连接层2：120 -> 84
        self.fc2 = nn.Linear(120, 84)
        # 全连接层3（输出层）：84 -> 10
        self.fc3 = nn.Linear(84, num_classes)
        # 激活函数
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()  # LeNet原始使用tanh，也可以使用ReLU
        
    def forward(self, x):
        # 输入: (batch_size, 1, 28, 28)
        x = self.tanh(self.conv1(x))  # (batch_size, 6, 28, 28)
        x = self.pool1(x)  # (batch_size, 6, 14, 14)
        x = self.tanh(self.conv2(x))  # (batch_size, 16, 10, 10)
        x = self.pool2(x)  # (batch_size, 16, 5, 5)
        x = x.view(x.size(0), -1)  # 展平: (batch_size, 16*5*5)
        x = self.tanh(self.fc1(x))  # (batch_size, 120)
        x = self.tanh(self.fc2(x))  # (batch_size, 84)
        x = self.fc3(x)  # (batch_size, 10)
        return x


def train_model(model, train_loader, criterion, optimizer, device, epoch):
    """训练一个epoch并返回训练准确率"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    
    # 计算平均损失和准确率
    avg_loss = running_loss / len(train_loader)
    train_acc = 100.0 * correct / total
    
    return avg_loss, train_acc


def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")
    
    # 超参数设置
    batch_size = 512
    learning_rate = 0.1
    num_epochs = 10
    num_classes = 10
    
    # 数据加载
    # 设置数据保存路径
    data_root = os.path.join(os.getcwd(), 'data')
    os.makedirs(data_root, exist_ok=True)
    
    # 数据转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        # MNIST数据已经是[0,1]范围，可以添加归一化
        # transforms.Normalize((0.1307,), (0.3081,))  # MNIST的均值和标准差
    ])
    
    # 加载训练集
    train_dataset = datasets.MNIST(
        root=data_root,
        train=True,
        transform=transform,
        download=True
    )
    
    # 加载测试集
    test_dataset = datasets.MNIST(
        root=data_root,
        train=False,
        transform=transform,
        download=True
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    print(f"training set: {len(train_dataset)}")
    print(f"testing set: {len(test_dataset)}")
    
    # 创建模型
    model = LeNet(num_classes=num_classes).to(device)
    print(f"\nmodel:")
    print(model)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # 训练循环
    print(f"\nstart to training，total {num_epochs} epoch...")
    print("=" * 60)
    
    for epoch in range(num_epochs):
        # 训练一个epoch
        train_loss, train_acc = train_model(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # 打印训练结果
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  training loss: {train_loss:.4f}")
        print(f"  training acc: {train_acc:.2f}%")
        print("-" * 60)
    
    print("\ndone！")
    
    # 在测试集上评估
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            test_total += target.size(0)
            test_correct += (predicted == target).sum().item()
    
    test_acc = 100.0 * test_correct / test_total
    print(f"\ntesting acc: {test_acc:.2f}%")

if __name__ == '__main__':
    main()

