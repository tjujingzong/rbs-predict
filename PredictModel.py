import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.model_zoo as model_zoo
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models.googlenet import GoogLeNet, model_urls
from torchvision.models.mobilenet import MobileNetV2
from torchvision.models.mobilenet import mobilenet_v2
from sklearn.metrics import mean_absolute_error
import numpy as np
import os
from sklearn.metrics import r2_score


# dfs遍历所有的碱基组合
def dfs(current, alphabet, char_map, layer):
    if len(current) == layer:
        index = len(char_map)
        one_hot = [0] * (4 ** layer)
        one_hot[index] = 1
        char_map[current] = one_hot
    else:
        for base in alphabet:
            dfs(current + base, alphabet, char_map, layer)


# 编码方式
def encoding(data, encode_type):
    if encode_type == 'one-hot':
        char_map = {
            'A': [1, 0, 0, 0],
            'T': [0, 1, 0, 0],
            'C': [0, 0, 1, 0],
            'G': [0, 0, 0, 1]
        }
        seq = []
        for sequence in data:
            tmp = [[], [], [], []]
            for item in sequence:
                item = item.upper()
                if item in char_map:
                    for i in range(4):
                        tmp[i].append(char_map[item][i])
                else:
                    for i in range(4):
                        tmp[i].append(0)
            seq.append(tmp)
        output = torch.tensor(seq, dtype=torch.float32)
    elif encode_type == 'triplet':
        alphabet = ['A', 'T', 'C', 'G']
        char_map = {}
        dfs("", alphabet, char_map, 3)
        seq = []
        for sequence in data:
            tmp = []
            for i in range(len(sequence) - 2):  # 仅对序列中的前n-2个碱基进行编码
                triplet = sequence[i:i + 3].upper()
                if triplet in char_map:
                    tmp.append(char_map[triplet])
                else:
                    tmp.append([0] * 64)  # 对于无效的三碱基组合，使用全零编码
            seq.append(tmp)
        output = torch.tensor(seq, dtype=torch.float32)
        output = output.permute(0, 2, 1)
    elif encode_type == 'dimer':
        alphabet = ['A', 'T', 'C', 'G']
        char_map = {}
        dfs("", alphabet, char_map, 2)
        seq = []
        for sequence in data:
            tmp = []
            for i in range(len(sequence) - 1):  # 对序列中的前n-1个碱基进行编码
                dimer = sequence[i:i + 2].upper()
                if dimer in char_map:
                    tmp.append(char_map[dimer])
                else:
                    tmp.append([0] * 16)  # 对于无效的二碱基组合，使用全零编码
            seq.append(tmp)
        output = torch.tensor(seq, dtype=torch.float32)
        output = output.permute(0, 2, 1)

    output = output.unsqueeze(1)
    return output.to(device)


# 读取数据
def read_data(path):
    dataset = pd.read_csv(path, header=None)
    x = dataset.iloc[:, 0]
    y = dataset.iloc[:, 1]
    l = len(x[0])  # 序列长度
    x_encoded = encoding(x, encoder)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    # 划分训练集和测试集 当random_state设置为一个固定的整数值时，每次运行train_test_split函数时，数据的分割结果将保持一致。
    x_train, x_test, y_train, y_test = train_test_split(x_encoded, y, test_size=0.2, random_state=43)

    x_train = x_train.to(device)
    x_test = x_test.to(device)
    y_train = y_train.to(device)
    y_test = y_test.to(device)

    return x_train, y_train, x_test, y_test, l


# WrapperGoogleNet
class WrapperGoogleNet(GoogLeNet):
    def __init__(self, *args, pretrained=False, **kwargs):
        super(WrapperGoogleNet, self).__init__(*args, **kwargs)
        if pretrained:
            # 下面的这部分代码在 pretrained=True 时加载预训练权重，并去掉了与辅助分类器（auxiliary classifiers）相关的权重
            state_dict = model_zoo.load_url(model_urls['googlenet'], progress=True)
            state_dict = {k: v for k, v in state_dict.items() if "aux" not in k}
            self.load_state_dict(state_dict, strict=False)


class GoogleNet(nn.Module):
    def __init__(self, num_classes, input_channels):
        super(GoogleNet, self).__init__()
        self.googlenet = WrapperGoogleNet(pretrained=True, aux_logits=False)
        # 更改输入通道数 ImageNet 数据集上训练的原始 GoogleNet 模型中的参数设置
        self.googlenet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        # 使用了一个预训练的 GoogleNet 模型，因此 self.googlenet.fc 的输入维度固定为 1024。
        self.googlenet.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        return self.googlenet(x)


class WrapperMobileNet(MobileNetV2):
    def __init__(self, *args, pretrained=False, **kwargs):
        super(WrapperMobileNet, self).__init__(*args, **kwargs)
        if pretrained:
            pretrained_model = mobilenet_v2(pretrained=True)
            self.load_state_dict(pretrained_model.state_dict(), strict=False)


class MobileNet(nn.Module):
    def __init__(self, num_classes, input_channels):
        super(MobileNet, self).__init__()
        self.mobilenet = WrapperMobileNet(pretrained=True)
        # 更改输入通道数 并与其他参数的设置与 MobileNetV2 中的一致。
        self.mobilenet.features[0][0] = nn.Conv2d(input_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                                                  bias=False)
        self.mobilenet.classifier[1] = nn.Linear(1280, num_classes)

    def forward(self, x):
        return self.mobilenet(x)


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(train_loader)


def evaluate(model, test_loader, criterion, device):  # 计算测试损失
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

    return running_loss / len(test_loader)


def get_predictions(model, dataset, device):
    model.eval()
    with torch.no_grad():
        inputs = dataset.tensors[0].to(device)
        outputs = model(inputs)
    return outputs.cpu().numpy()


if __name__ == "__main__":
    encoders = ['one-hot', 'triplet', 'dimer']
    encoder = encoders[2]
    data_names = ['rbs', 'promoter', 'rbs1', 'promoter1']
    data_name = data_names[2]
    data_path = r'./data/' + data_name + '-data.csv'
    model_names = ['GoogleNet', 'MobileNet']
    model_name = model_names[1]
    learning_rate = 0.001
    batch_size = 64
    num_epochs = 70
    num_classes = 1  # 输出类别数
    input_channels = 1  # 输入通道数 一般图像为3，这里由于是序列，所以为1
    model_path = 'model/' + data_name + '-' + model_name + '_' + encoder + '.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('current device is:{}'.format(device))
    x_train, y_train, x_test, y_test, width = read_data(data_path)

    print('sequence length(width) is:{}'.format(width))
    print('x_data shape is:{}'.format(x_train.shape))
    # 构建数据集和数据加载器
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型、损失函数、优化器
    if model_name == 'GoogleNet':
        model = GoogleNet(num_classes, input_channels).to(device)
    elif model_name == 'MobileNet':
        model = MobileNet(num_classes, input_channels).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if os.path.exists(model_path):  # 如果模型已存在，加载已有模型
        print("Loading pre-trained model...")
        model.load_state_dict(torch.load(model_path))
        model.eval()
    else:
        train_losses = []
        test_losses = []
        # 训练模型
        for epoch in range(num_epochs):
            train_loss = train(model, train_loader, criterion, optimizer, device)
            test_loss = evaluate(model, test_loader, criterion, device)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        pic_name1 = 'pic/' + data_name + '-' + model_name + '_' + encoder + '_loss' + '.png'
        plt.savefig(pic_name1)
        torch.save(model.state_dict(), model_path)

    model.eval()  # 切换到评估模式
    train_preds = get_predictions(model, train_dataset, device)
    test_preds = get_predictions(model, test_dataset, device)

    pearson_corr = np.corrcoef(y_test.cpu().numpy().flatten(), test_preds.flatten())[0, 1]  # 返回的是一个矩阵
    mae = mean_absolute_error(y_test.cpu().numpy().flatten(), test_preds.flatten())
    r2 = r2_score(y_test.cpu().numpy().flatten(), test_preds.flatten())

    side_len = 100
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.scatter(y_train.cpu().numpy(), train_preds, alpha=0.5)
    ax1.set_xlabel('True Values')
    ax1.set_ylabel('Predictions')
    ax1.set_title('Train Set')
    ax1.set_aspect('equal', 'box')  # 添加这一行以使横纵坐标尺寸相同
    ax1.axis('square')  # 添加这一行以使子图呈正方形
    ax1.set_xlim(0, side_len)
    ax1.set_ylim(0, side_len)

    ax2.scatter(y_test.cpu().numpy(), test_preds, alpha=0.5)
    ax2.set_xlabel('True Values')
    ax2.set_ylabel('Predictions')
    ax2.set_title('Test Set')
    ax2.set_aspect('equal', 'box')  # 添加这一行以使横纵坐标尺寸相同
    ax2.axis('square')  # 添加这一行以使子图呈正方形
    ax2.set_xlim(0, side_len)
    ax2.set_ylim(0, side_len)

    pic_name2 = 'pic/' + data_name + '-' + model_name + '_' + encoder + '_scatter' + '.png'
    plt.savefig(pic_name2)
    # print(model)

    with open("log.txt", "a") as log_file:
        log_file.write(f"DataSet: {data_name}\n")
        log_file.write(f"Model: {model_name}\n")
        log_file.write(f"Encoder: {encoder}\n")
        log_file.write(f"Num epochs: {num_epochs}\n")
        log_file.write(f"Learning rate: {learning_rate}\n")
        log_file.write(f"Batch size: {batch_size}\n")
        log_file.write(f"Mean Absolute Error: {mae:.4f}\n")
        log_file.write(f"Pearson Correlation Coefficient: {pearson_corr:.4f}\n")
        log_file.write(f"R^2 Score: {r2:.4f}\n")  # 添加 R^2 值
        log_file.write("\n")
