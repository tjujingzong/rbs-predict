import csv

# 打开csv文件
with open("data/di_prop.csv", newline="") as csvfile:
    # 创建一个csv阅读器对象
    reader = csv.reader(csvfile)
    # 读取第一行，作为列名
    headers = next(reader)
    # 去除第一列
    headers.pop(0)
    # 创建一个空字典，用于存储每一列的数值
    data = {}
    # 遍历每一列名，初始化字典的键和值
    for header in headers:
        data[header] = []
    # 遍历剩余的每一行，将数值添加到对应的列表中
    row_count = 0
    for row in reader:
        row.pop(0)
        if row_count == 11:  # 最多读取90行
            break
        for i, value in enumerate(row):
            # 将字符串转换为浮点数
            value = float(value)
            # 获取对应的列名
            header = headers[i]
            # 将数值添加到列表中
            data[header].append(value)
        row_count += 1
# 去除data key中的空格
new_data = {}
# 遍历data中的每个key和value
for key, value in data.items():
    # 使用replace()方法将key中的空格替换为空字符
    new_key = key.replace(" ", "")
    # 将新的key和value添加到新的字典中
    new_data[new_key] = value
# 打印新的字典
print(new_data)
