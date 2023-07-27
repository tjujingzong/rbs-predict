import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

device = torch.device('cpu')
encoders = ['one-hot', 'triplet', 'dimer', 'dimer-1', 'combined']
encoder = encoders[0]
data_names = ['RBS-232', 'RBS-317', 'RBS-417']
data_name = data_names[0]
data_path = r'./data/' + data_name + '-data.csv'
model_names = ['GoogleNet', 'MobileNet', 'SVM']
model_name = model_names[1]
learning_rate = 0.001
batch_size = 64
num_epochs = 70
num_classes = 1  # 输出类别数
input_channels = 1  # 输入通道数 一般图像为3，这里由于是序列，所以为1
model_path = 'model/' + data_name + '-' + model_name + '_' + encoder + '.pth'


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
            'A': 1,
            'T': 2,
            'C': 3,
            'G': 4
        }
        seq = []
        for sequence in data:
            tmp = []
            for base in sequence:
                tmp.append(char_map[base])
            seq.append(tmp)
        output = torch.tensor(seq, dtype=torch.float32)
    elif encode_type == 'dimer':
        char_map = {
            'GG': [-0.01, -1.78, 3.32, 0.3, 12.1, 32.0, -11.1, -12.2, -29.7, -3.26, 0.17],
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
    elif encode_type == 'dimer-1':
        alphabet = ['A', 'T', 'C', 'G']
        char_map = {}
        dfs("", alphabet, char_map, 2)
        seq = []
        for sequence in data:
            tmp = []
            for i in range(len(sequence) - 1):  # 仅对序列中的前n-2个碱基进行编码
                triplet = sequence[i:i + 2].upper()
                if triplet in char_map:
                    tmp.append(char_map[triplet])
                else:
                    tmp.append([0] * 16)  # 对于无效的三碱基组合，使用全零编码
            seq.append(tmp)
        output = torch.tensor(seq, dtype=torch.float32)
        output = output.permute(0, 2, 1)
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
    elif encode_type == 'combined':
        one_hot_encoded = encoding(data, 'one-hot')

        dimer_encoded = encoding(data, 'dimer')
        zero_tensor = torch.zeros_like(dimer_encoded)
        zero_tensor = zero_tensor[..., :1]
        dimer_encoded = torch.cat([dimer_encoded, zero_tensor], dim=-1)

        dimer_1_encoded = encoding(data, 'dimer-1')
        zero_tensor = torch.zeros_like(dimer_1_encoded)
        zero_tensor = zero_tensor[..., :1]
        dimer_1_encoded = torch.cat([dimer_1_encoded, zero_tensor], dim=-1)

        triplet_encoded = encoding(data, 'triplet')
        zero_tensor = torch.zeros_like(triplet_encoded)
        zero_tensor = zero_tensor[..., :2]
        triplet_encoded = torch.cat([triplet_encoded, zero_tensor], dim=-1)

        secondary_structure = get_secondary_structure(data_path)

        output = torch.cat([one_hot_encoded, dimer_encoded, dimer_1_encoded, triplet_encoded, secondary_structure],
                           dim=2)
        print(output.shape)
        return output.to(device)
    # output = output.unsqueeze(1)
    return output.to(device)


def get_secondary_structure(data_path):
    dataset = pd.read_csv(data_path, header=None)
    # dataset = dataset.iloc[1:, :]  # 去除第一行
    y = dataset.iloc[:, -1]  # 取出最后一列
    dic = {'(': -1, '.': 0, ')': 1}
    y = y.apply(lambda x: [dic[i] for i in x])  # 将每一行的序列依次转换为数字
    y = y.reset_index(drop=True)  # 重置索引
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # 转换为torch张量
    y = y.unsqueeze(1)
    print(y.shape)
    return y.to(device)


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


if __name__ == "__main__":
    print('current device is:{}'.format(device))
    x_train, y_train, x_test, y_test, width = read_data(data_path)
    print('x_data shape is:{}'.format(x_train.shape))
    y_test = y_test.flatten()
    y_train = y_train.flatten()
    # num_samples = x_train.size(0)  # Get the number of samples (232 in this case)
    # num_features = x_train.size(2) * x_train.size(3)  # Calculate the total number of features (96 * 30)
    # x_train_2d = x_train.view(num_samples, num_features)
    #
    # num_samples = x_test.size(0)  # Get the number of samples (232 in this case)
    # num_features = x_test.size(2) * x_test.size(3)  # Calculate the total number of features (96 * 30)
    # x_test_2d = x_test.view(num_samples, num_features)

    svr_regressor = SVR()
    svr_regressor.fit(x_train, y_train)
    y_pred = svr_regressor.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    pearson_corr, _ = pearsonr(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("SVR MAE:", mae)
    print("SVR 皮尔森相关系数:", pearson_corr)
    print("SVR R2:", r2)

    elasticnet_regressor = ElasticNet()
    elasticnet_regressor.fit(x_train, y_train)
    y_pred = elasticnet_regressor.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    pearson_corr, _ = pearsonr(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("ElasticNet Regression MAE:", mae)
    print("ElasticNet Regression Pearson correlation:", pearson_corr)
    print("ElasticNet Regression R2:", r2)

    # Assuming x_train, y_train, x_test, y_test are already defined
    knn_regressor = KNeighborsRegressor()
    knn_regressor.fit(x_train, y_train)
    y_pred = knn_regressor.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    pearson_corr, _ = pearsonr(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("K-Nearest Neighbors Regression MAE:", mae)
    print("K-Nearest Neighbors Regression Pearson correlation:", pearson_corr)
    print("K-Nearest Neighbors Regression R2:", r2)
