import pandas as pd
import matplotlib.pyplot as plt

# Read CSV file
data_name = 'RBS-317'
data_path = 'data/' + data_name + '.csv'
data = pd.read_csv(data_path)

# Extract the first column data
column_data = data.iloc[:, 1]

# 在每个矩形上方添加数量标签
n, bins, patches = plt.hist(column_data, bins=20, edgecolor='black', alpha=0.7)
hist = n
for x, y in zip(bins, hist):
    # 如果y不为0
    if y != 0:
        plt.text(x + 50, y + 10, '%.0f' % y, ha='center', va='top', fontsize=10)
        # 调整数字的位置，使其向右偏移

plt.xlabel('Strength')
plt.ylabel('Frequency')
plt.title(data_name + ' Data Distribution')

# Save image (optional)
plt.savefig('pic/' + data_name + '-histogram.png')

# Show image (optional)
plt.show()
