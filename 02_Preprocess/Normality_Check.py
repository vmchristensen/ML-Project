import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path

# Load data
script_dir = Path(__file__).parent
data_path = script_dir.parent / "01_Data" / "preprocessed_train.csv"
data = pd.read_csv(data_path)

# Define your numerical features
if 'pagerank' in data.columns:
    numerical_features = ['degree', 'closeness', 'betweenness', 'pagerank', 'harmonic', 'eigenvector', 'eccentricity']
else:
    numerical_features = ['degree', 'eccentricity']

print(f"Checking normality for features: {numerical_features}")

SAMPLE_SIZE = 5000

# Separate data by class
data_class_0 = data[data['target'] == 0]
data_class_1 = data[data['target'] == 1]

# --- Visual Inspection with Sampling ---
for feature in numerical_features:
    print(f"Processing feature: {feature}")
    plt.figure(figsize=(12, 10)) 

    # --- Histograms ---
    sampled_c0_hist = data_class_0[feature].dropna()
    if len(sampled_c0_hist) > SAMPLE_SIZE:
        sampled_c0_hist = sampled_c0_hist.sample(SAMPLE_SIZE, random_state=42)
    
    sampled_c1_hist = data_class_1[feature].dropna()
    if len(sampled_c1_hist) > SAMPLE_SIZE:
        sampled_c1_hist = sampled_c1_hist.sample(SAMPLE_SIZE, random_state=42)

    plt.subplot(2, 2, 1)
    if not sampled_c0_hist.empty:
        sns.histplot(sampled_c0_hist, kde=True, color='blue', label='Class 0 (Not Root)', bins=50)
    plt.title(f'Sampled Hist of {feature} (Class 0)')
    plt.legend()

    plt.subplot(2, 2, 2)
    if not sampled_c1_hist.empty:
        sns.histplot(sampled_c1_hist, kde=True, color='red', label='Class 1 (Root)', bins=50)
    plt.title(f'Sampled Hist of {feature} (Class 1)')
    plt.legend()

    # --- Q-Q Plots ---
    sampled_c0_qq = data_class_0[feature].dropna()
    if len(sampled_c0_qq) > SAMPLE_SIZE:
        sampled_c0_qq = sampled_c0_qq.sample(SAMPLE_SIZE, random_state=42)
    
    sampled_c1_qq = data_class_1[feature].dropna()
    if len(sampled_c1_qq) > SAMPLE_SIZE:
        sampled_c1_qq = sampled_c1_qq.sample(SAMPLE_SIZE, random_state=42)

    plt.subplot(2, 2, 3)
    if len(sampled_c0_qq) > 1:
        stats.probplot(sampled_c0_qq, dist="norm", plot=plt)
    plt.title(f'Sampled Q-Q Plot of {feature} (Class 0)')

    plt.subplot(2, 2, 4)
    if len(sampled_c1_qq) > 1:
        stats.probplot(sampled_c1_qq, dist="norm", plot=plt)
    plt.title(f'Sampled Q-Q Plot of {feature} (Class 1)')
    
    plt.suptitle(f'Distribution and Q-Q Plots for {feature} by Class (Sampled N={SAMPLE_SIZE})')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# --- Statistical Tests ---
run_statistical_tests = False

if run_statistical_tests:
    print("\n--- Shapiro-Wilk Normality Test (on sampled data) ---")
    print("Note: Low p-value (e.g., < 0.05) suggests data is NOT normally distributed.")
    
    for feature in numerical_features:
        # Test for Class 0
        sample_class_0_test = data_class_0[feature].dropna()
        if len(sample_class_0_test) > SAMPLE_SIZE:
            sample_class_0_test = sample_class_0_test.sample(SAMPLE_SIZE, random_state=42)
        
        if len(sample_class_0_test) >= 3 and len(sample_class_0_test) <= 5000:
            stat_0, p_0 = stats.shapiro(sample_class_0_test)
            print(f'Class 0, {feature}: Statistics={stat_0:.3f}, p-value={p_0:.3g}')
        elif len(sample_class_0_test) > 0:
            print(f'Class 0, {feature}: Sample size {len(sample_class_0_test)} not ideal for Shapiro-Wilk. Consider K-S test or visual inspection.')
        else:
            print(f'Class 0, {feature}: No data points for test.')


        # Test for Class 1
        sample_class_1_test = data_class_1[feature].dropna()
        if len(sample_class_1_test) > SAMPLE_SIZE:
            sample_class_1_test = sample_class_1_test.sample(SAMPLE_SIZE, random_state=42)

        if len(sample_class_1_test) >= 3 and len(sample_class_1_test) <= 5000:
            stat_1, p_1 = stats.shapiro(sample_class_1_test)
            print(f'Class 1, {feature}: Statistics={stat_1:.3f}, p-value={p_1:.3g}\n')
        elif len(sample_class_1_test) > 0:
            print(f'Class 1, {feature}: Sample size {len(sample_class_1_test)} not ideal for Shapiro-Wilk. Consider K-S test or visual inspection.\n')
        else:
            print(f'Class 1, {feature}: No data points for test.\n')