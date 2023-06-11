import pandas as pd
import torch
import torch.nn as nn
from torchvision.models.mobilenet import mobilenet_v2


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
        char_map = {'GG': [-0.01, -1.78, 3.32, 0.3, 12.1, 32.0, -11.1, -12.2, -29.7, -3.26, 0.17],
                    'GA': [0.07, -1.7, 3.38, 1.3, 9.4, 32.0, -14.2, -13.3, -35.5, -2.35, 0.1],
                    'GC': [0.07, -1.39, 3.22, 0.0, 6.1, 35.0, -16.9, -14.2, -34.9, -3.42, 0.26],
                    'GT': [0.23, -1.43, 3.24, 0.8, 4.8, 32.0, -13.8, -10.2, -26.2, -2.24, 0.27],
                    'AG': [-0.04, -1.5, 3.3, 0.5, 8.5, 30.0, -14.0, -7.6, -19.2, -2.08, 0.08],
                    'AA': [-0.08, -1.27, 3.18, -0.8, 7.0, 31.0, -13.7, -6.6, -18.4, -0.93, 0.04],
                    'AC': [0.23, -1.43, 3.24, 0.8, 4.8, 32.0, -13.8, -10.2, -26.2, -2.24, 0.14],
                    'AT': [-0.06, -1.36, 3.24, 1.1, 7.1, 33.0, -15.4, -5.7, -15.5, -1.1, 0.14],
                    'CG': [0.3, -1.89, 3.3, -0.1, 12.1, 27.0, -15.6, -8.0, -19.4, -2.36, 0.35],
                    'CA': [0.11, -1.46, 3.09, 1.0, 9.9, 31.0, -14.4, -10.5, -27.8, -2.11, 0.21],
                    'CC': [-0.01, -1.78, 3.32, 0.3, 8.7, 32.0, -11.1, -12.2, -29.7, -3.26, 0.49],
                    'CT': [-0.04, -1.5, 3.3, 0.5, 8.5, 30.0, -14.0, -7.6, -19.2, -2.08, 0.52],
                    'TG': [0.11, -1.46, 3.09, 1.0, 9.9, 31.0, -14.4, -7.6, -19.2, -2.11, 0.34],
                    'TA': [-0.02, -1.45, 3.26, -0.2, 10.7, 32.0, -16.0, -8.1, -22.6, -1.33, 0.21],
                    'TC': [0.07, -1.7, 3.38, 1.3, 9.4, 32.0, -14.2, -10.2, -26.2, -2.35, 0.48],
                    'TT': [-0.08, -1.27, 3.18, -0.8, 7.0, 31.0, -13.7, -6.6, -18.4, -0.93, 0.44]}
        seq = []
        for sequence in data:
            tmp = []
            for i in range(len(sequence) - 1):  # 对序列中的前n-1个碱基进行编码
                dimer = sequence[i:i + 2].upper()
                if dimer in char_map:
                    tmp.append(char_map[dimer])
                else:
                    tmp.append([0] * 11)  # 对于无效的二碱基组合，使用全零编码
            seq.append(tmp)
        output = torch.tensor(seq, dtype=torch.float32)
        output = output.permute(0, 2, 1)

    output = output.unsqueeze(1)
    return output.to(device)


# 读取数据
def read_data(path):
    dataset = pd.read_csv(path, header=None)
    x = dataset.iloc[:, 0]
    # 去除第一行
    x = x.drop(labels=0, axis=0)
    return x


class MobileNet(nn.Module):
    def __init__(self, num_classes, input_channels):
        super(MobileNet, self).__init__()
        # Use the pretrained model as a base and modify the input and output layers
        self.mobilenet = mobilenet_v2(pretrained=True)
        self.mobilenet.features[0][0] = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.mobilenet.classifier[1] = nn.Linear(1280, num_classes)

    def forward(self, x):
        return self.mobilenet(x)


class EnsembleNet_M(nn.Module):
    def __init__(self, num_classes, input_channels):
        super(EnsembleNet_M, self).__init__()
        self.mobilenet_dimer = MobileNet(num_classes, input_channels)
        self.mobilenet_triplet = MobileNet(num_classes, input_channels)
        # Define an adaptive average pooling layer with output size 1x1
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1280 * 2, 1)

    def forward(self, x_dimer, x_triplet):
        # Extract the features from both mobilenets and pool them
        features_dimer = self.pool(self.mobilenet_dimer.mobilenet.features(x_dimer))
        features_triplet = self.pool(self.mobilenet_triplet.mobilenet.features(x_triplet))
        # Concatenate the features along the channel dimension and flatten them
        features = torch.flatten(torch.cat([features_dimer, features_triplet], dim=1), 1)
        # Use the fully connected layer to get the final output
        out = self.fc(features)
        return out


if __name__ == "__main__":

    data_names = ['RBS-317']
    data_name = data_names[0]
    model_names = ['GoogleNet', 'MobileNet']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_path = r'./data/seqs.csv'
    seqs = read_data(data_path)

    model_paths = ['model/RBS-317-MobileNet_triplet.pth', 'model/RBS-317-MobileNet_dimer.pth',
                   'model/RBS-317-EnsembleNet-M.pth']

    for model_path in model_paths:
        if model_path.find('Ensemble') == -1:
            model = MobileNet(num_classes=1, input_channels=1)  # create an instance of the same model class
        else:
            model = EnsembleNet_M(num_classes=1, input_channels=1)
        model.to(device)  # move the model to the device
        model.load_state_dict(torch.load(model_path))  # load the parameters from the file
        model.eval()  # set the model to evaluation mode
        if model_path.find('triplet') != -1:
            encoder = 'triplet'
        elif model_path.find('dimer') != -1:
            encoder = 'dimer'
        result = []
        for sequence in seqs:
            if model_path.find('Ensemble') == -1:
                encoded_sequence = encoding([sequence], encode_type=encoder)  # encode the sequence into a tensor
                intensity = model.forward(encoded_sequence)  # pass the tensor to the model and get the output
            else:
                encoded_sequence1 = encoding([sequence], encode_type='dimer')
                encoded_sequence2 = encoding([sequence], encode_type='triplet')
                intensity = model.forward(encoded_sequence1, encoded_sequence2)
            result.append(intensity.item())
            print(intensity.item())

    # 保存预测结果到data_path中新的一列
    df = pd.read_csv(data_path)
    df[model_path] = result
    df.to_csv(data_path, index=False)
