import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from models.SVEFormer import MultiheadAttention
from models.fourierSVD import get_svd_EF, overlap_rate, conv_svd
import time
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

# 定义卷积、批归一化和激活函数
def getConvBn1Relu(head_dim):
    conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    conv2 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1, bias=False)
    bn1 = nn.BatchNorm2d(head_dim)
    relu = nn.ReLU(inplace=True)
    return conv1, conv2, bn1, relu

# 定义带有自注意力的残差块
class ResidualBlockWithAttention(nn.Module):
    def __init__(self, Vt, in_channels, out_channels, stride=1):
        super(ResidualBlockWithAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        self.self_attention = MultiheadAttention(Vt, input_dim=1024, hidden_dim=out_channels, num_heads=4)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.downsample(identity)
        out = self.relu(out)

        # 展平空间维度以适应注意力机制
        B, C, H, W = out.size()
        out = out.view(B, C, -1).permute(2, 0, 1).contiguous()
        # 应用自注意力
        out = self.self_attention(out.transpose(0, 1)).transpose(0, 1)

        # 恢复原始形状
        out = out.permute(1, 2, 0).contiguous().view(B, C, H, W)
        return out

# 定义带有自注意力的ResNet模型
class ResNetWithAttention(nn.Module):
    def __init__(self, Vt, block, num_blocks, num_classes=10):
        super(ResNetWithAttention, self).__init__()
        self.in_channels = 64
        self.out_channels = 256
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(Vt, block, self.out_channels, num_blocks[0], stride=1)
        self.layer2 = self.make_layer(Vt, block, self.out_channels, num_blocks[1], stride=1)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.out_channels, num_classes)
        self.Vt = Vt

    def make_layer(self, Vt, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(Vt, self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x, return_features=False):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.avg_pool(out)
        features = out.view(out.size(0), -1)  # 提取特征
        out = self.fc(features)
        if return_features:
            return out, features
        return out

# 检查是否有可用的GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 数据预处理
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR-10 的均值和标准差
])

batch_size = 16
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

start = time.perf_counter()

# 初始化傅里叶基矩阵 Vt
head_dim = 256
num_head = 4
k = 64
conv1, conv2, bn1, relu = getConvBn1Relu(head_dim)
S = torch.zeros((1024, 1024)).to(device)

pre_indices = torch.zeros((1024, 1)).to(device)
indices = torch.zeros((1024, 1)).to(device)
j = 0

stable = 0
for i, data in enumerate(trainloader, 0):
    j += 1
    inputs, labels = data
    inputs = conv1(inputs)
    inputs = conv2(inputs)
    inputs = bn1(inputs)
    inputs = relu(inputs)
    B, C, H, W = inputs.shape
    out = inputs.view(B, C, -1).permute(2, 0, 1).contiguous().to(device)
    N, B, C = out.shape
    out = out.reshape(B * num_head, H * W, C // num_head)
    indices, lamda = get_svd_EF(out.transpose(-1, -2), S, k, i + 1)
    if overlap_rate(indices.to(device), pre_indices.to(device), k, 0.9) or i == 781:
        stable = stable + 1
        print('第j个batch找到傅里叶基', i)
        Lambda, Vt = conv_svd(lamda, device)
        Vt = Vt.to(device)
        Vt = Vt[:, indices]
        break
    else:
        S = lamda
        pre_indices = indices
        stable = 0

# 定义模型
resnet_with_attention = ResNetWithAttention(Vt.transpose(-1, -2), ResidualBlockWithAttention, [2, 2, 2, 2]).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet_with_attention.parameters(), lr=1e-4)  # 使用 Adam 优化器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

print("Training Begin")

# 定义测试函数
def evaluate_model(model, dataloader, device):
    model.eval()  # 设置为评估模式
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# 定义提取特征和可视化的函数
def visualize_clusters(model, dataloader, device, epoch, num_samples=1000):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for data in dataloader:
            images, lbls = data
            images = images.to(device)
            outputs, feats = model(images, return_features=True)
            features.append(feats.cpu().numpy())
            labels.append(lbls.numpy())
            if len(features) * images.size(0) >= num_samples:
                break
    features = np.concatenate(features, axis=0)[:num_samples]
    labels = np.concatenate(labels, axis=0)[:num_samples]

    # 使用 t-SNE 进行降维
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)

    # 绘制散点图
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab10', alpha=0.6)
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title(f'CIFAR-10 Feature Clusters at Epoch {epoch}')
    plt.xlabel('TSNE Component 1')
    plt.ylabel('TSNE Component 2')
    plt.grid(True)
    plt.savefig(f'sve_cluster_epoch_{epoch}.png')  # 保存图像
    plt.close()
    print(f'已保存第 {epoch} 个epoch的聚类可视化图像.')

# 训练循环
for epoch in range(30):  # 根据需要更改epoch数量
    resnet_with_attention.train()  # 设置为训练模式
    running_loss = 0.0
    total_train = 0
    correct_train = 0

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)  # 将数据移动到GPU
        optimizer.zero_grad()
        outputs = resnet_with_attention(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(resnet_with_attention.parameters(), max_norm=0.5)  # 梯度裁剪
        optimizer.step()

        # 统计训练损失和准确率
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

        if i % 200 == 199:  # 每200个batch打印一次
            print('[Epoch %d, Batch %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0

    # 计算训练集准确率
    train_accuracy = 100 * correct_train / total_train
    print(f'Epoch {epoch + 1} Training Accuracy: {train_accuracy:.3f} %')

    # 测试模型
    test_accuracy = evaluate_model(resnet_with_attention, testloader, device)
    print(f'Epoch {epoch + 1} Test Accuracy: {test_accuracy:.3f} %')

    # 每5个epoch进行一次聚类可视化
    if (epoch + 1) % 5 == 0:
        visualize_clusters(resnet_with_attention, testloader, device, epoch + 1)

    scheduler.step()

end = time.perf_counter()
print('Finished Training')
print("Time cost:", end - start)