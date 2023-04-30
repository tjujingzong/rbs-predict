import pandas as pd
import matplotlib.pyplot as plt

# Read CSV file
data_name = 'rbs'
data_path = 'data/' + data_name + '-data.csv'
data = pd.read_csv(data_path)

# Extract the first column data
column_data = data.iloc[:, 1]

# Plot histogram
plt.hist(column_data, bins=10, edgecolor='black', alpha=0.7)
plt.xlabel('Strength')
plt.ylabel('Frequency')
plt.title(data_name + ' Data Distribution')

# Save image (optional)
plt.savefig('pic/' + data_name + '-histogram.png')

# Show image (optional)
plt.show()
