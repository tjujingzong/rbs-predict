# 读取data/ga_RBS_seq(1).txt文件，将其转换为csv文件


import pandas as pd


def txt_to_csv():
    with open('data/ga_RBS_seq(1).txt', 'r') as f:
        data = f.readlines()
    data = [i.strip() for i in data]
    data = pd.DataFrame(data)
    # 添加列名
    data.columns = ['seq']
    data.to_csv('data/ga_RBS_seq(1).csv', index=None)


txt_to_csv()
