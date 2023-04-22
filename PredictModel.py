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
import numpy as np


def dfs(current, alphabet, char_map):
    if len(current) == 3:
        index = len(char_map)
        one_hot = [0] * 64
        one_hot[index] = 1
        char_map[current] = one_hot
    else:
        for base in alphabet:
            dfs(current + base, alphabet, char_map)


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
    if encode_type == 'triplet':
        alphabet = ['A', 'T', 'C', 'G']
        char_map = {}
        dfs("", alphabet, char_map)
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

    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(x_encoded, y, test_size=0.2, random_state=41)

    x_train = x_train.to(device)
    x_test = x_test.to(device)
    y_train = y_train.to(device)
    y_test = y_test.to(device)

    return x_train, y_train, x_test, y_test, l


class CustomGoogLeNet(GoogLeNet):
    def __init__(self, *args, pretrained=False, **kwargs):
        super(CustomGoogLeNet, self).__init__(*args, **kwargs)
        if pretrained:
            # 下面的这部分代码在 pretrained=True 时加载预训练权重，并去掉了与辅助分类器（auxiliary classifiers）相关的权重
            state_dict = model_zoo.load_url(model_urls['googlenet'], progress=True)
            state_dict = {k: v for k, v in state_dict.items() if "aux" not in k}
            self.load_state_dict(state_dict, strict=False)

    def _transform_input(self, x):
        if self.transform_input:
            x = x.clone()
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            # 其中，0.229 和 0.485 分别是 ImageNet 数据集上的 RGB 通道的标准化参数（mean 和 std）。这里只处理第一个通道，因为输入数据只有一个通道。然后将处理后的结果赋值给 x。
            x = x_ch0
        return x


class RSBGoogLeNet(nn.Module):
    def __init__(self, num_classes, input_channels):
        super(RSBGoogLeNet, self).__init__()
        self.googlenet = CustomGoogLeNet(pretrained=True, aux_logits=False)
        # 更改输入通道数
        self.googlenet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.googlenet.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        return self.googlenet(x)


class CustomMobileNetV2(MobileNetV2):
    def __init__(self, *args, pretrained=False, **kwargs):
        super(CustomMobileNetV2, self).__init__(*args, **kwargs)
        if pretrained:
            pretrained_model = mobilenet_v2(pretrained=True)
            self.load_state_dict(pretrained_model.state_dict(), strict=False)


class RSBMobileNetV2(nn.Module):
    def __init__(self, num_classes, input_channels):
        super(RSBMobileNetV2, self).__init__()
        self.mobilenet = CustomMobileNetV2(pretrained=True)
        # 更改输入通道数
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
    encoder = 'triplet'
    data_names = ['promoter', 'rbs']
    data_name = data_names[1]
    data_path = r'./data/' + data_name + '-data.csv'
    model_names = ['RSBGoogLeNet', 'RSBMobileNetV2']
    modelname = model_names[1]
    learning_rate = 0.001
    batch_size = 64
    num_epochs = 80
    if encoder == 'one-hot':
        height = 4
    elif encoder == 'triplet':
        height = 64
    num_classes = 1  # 输出类别数
    input_channels = 1  # 输入通道数 一般图像为3，这里由于是序列，所以为1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('current device is:{}'.format(device))
    x_train, y_train, x_test, y_test, width = read_data(data_path)

    print('sequence length(width) is:{}'.format(width))
    print('x_data shape is:{}'.format(x_train.shape))
    print('height is:{}'.format(height))
    # 构建数据集和数据加载器
    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型、损失函数、优化器
    if modelname == 'RSBGoogLeNet':
        model = RSBGoogLeNet(num_classes, input_channels).to(device)
    elif modelname == 'RSBMobileNetV2':
        model = RSBMobileNetV2(num_classes, input_channels).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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
    pic_name1 = 'pic/' + data_name + '-' + modelname + '_' + encoder + '_loss' + '.png'
    plt.savefig(pic_name1)

    train_preds = get_predictions(model, train_dataset, device)
    test_preds = get_predictions(model, test_dataset, device)
    pearson_corr = np.corrcoef(y_test.cpu().numpy().flatten(), test_preds.flatten())[0, 1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.scatter(y_train.cpu().numpy(), train_preds, alpha=0.5)
    ax1.set_xlabel('True Values')
    ax1.set_ylabel('Predictions')
    ax1.set_title('Train Set')
    ax1.set_aspect('equal', 'box')  # 添加这一行以使横纵坐标尺寸相同
    ax1.axis('square')  # 添加这一行以使子图呈正方形

    ax2.scatter(y_test.cpu().numpy(), test_preds, alpha=0.5)
    ax2.set_xlabel('True Values')
    ax2.set_ylabel('Predictions')
    ax2.set_title('Test Set')
    ax2.set_aspect('equal', 'box')  # 添加这一行以使横纵坐标尺寸相同
    ax2.axis('square')  # 添加这一行以使子图呈正方形

    pic_name2 = 'pic/' + data_name + '-' + modelname + '_' + encoder + '_scatter' + '.png'
    plt.savefig(pic_name2)

    model_name = 'model/' + data_name + modelname + '_' + encoder + '.pth'
    torch.save(model.state_dict(), model_name)

    with open("log.txt", "a") as log_file:
        log_file.write(f"DataSet: {data_name}\n")
        log_file.write(f"Model: {modelname}\n")
        log_file.write(f"Encoder: {encoder}\n")
        log_file.write(f"Num epochs: {num_epochs}\n")
        log_file.write(f"Learning rate: {learning_rate}\n")
        log_file.write(f"Batch size: {batch_size}\n")
        log_file.write(f"Pearson Correlation Coefficient: {pearson_corr:.4f}\n")
        log_file.write("\n")
