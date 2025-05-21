import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load data
script_dir = Path(__file__).parent
data_path = script_dir.parent / "01_Data" / "preprocessed_train.csv"
data = pd.read_csv(data_path)

# Use full list of numerical features
numerical_features = ['degree', 'closeness', 'betweenness', 'pagerank', 'harmonic', 'eigenvector', 'eccentricity']

# Split data by class
data_class_0 = data[data['target'] == 0]
data_class_1 = data[data['target'] == 1]

SAMPLE_SIZE = 5000
fig, axes = plt.subplots(nrows=7, ncols=2, figsize=(14, 24))  # wider & taller

fig.suptitle('Histograms of Numerical Features by Class (Sampled)', fontsize=20)

for i, feature in enumerate(numerical_features):
    # Sample class 0
    values_0 = data_class_0[feature].dropna()
    if len(values_0) > SAMPLE_SIZE:
        values_0 = values_0.sample(SAMPLE_SIZE, random_state=42)

    # Sample class 1
    values_1 = data_class_1[feature].dropna()
    if len(values_1) > SAMPLE_SIZE:
        values_1 = values_1.sample(SAMPLE_SIZE, random_state=42)

    # Plot class 0
    sns.histplot(values_0, kde=True, ax=axes[i, 0], color='blue', bins=50)
    axes[i, 0].set_title(f'{feature} (Class 0 - Not Root)', fontsize=11)
    axes[i, 0].tick_params(axis='x', rotation=30)

    # Plot class 1
    sns.histplot(values_1, kde=True, ax=axes[i, 1], color='red', bins=50)
    axes[i, 1].set_title(f'{feature} (Class 1 - Root)', fontsize=11)
    axes[i, 1].tick_params(axis='x', rotation=30)

# Adjust spacing
plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=3.0)
plt.savefig("all_feature_histograms_by_class_readable.png", dpi=300)
plt.show()
