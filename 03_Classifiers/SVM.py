import pandas as pd 
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupKFold
from pathlib import Path
from sklearn.pipeline import Pipeline

# Load data
script_dir = Path(__file__).parent
data_path = script_dir.parent / "01_Data" / "preprocessed_train.csv"
data = pd.read_csv(data_path)
data = data.sample(100000, random_state=42)

# Define features and target
categorical_features = []
numerical_features = ['degree', 'closeness', 'betweenness', 'pagerank', 'harmonic', 'eigenvector', 'eccentricity']

# Preprocessing: One-hot encode language, scale numeric features (important for SVM)
preprocessor = ColumnTransformer(
    transformers=[
        ('lang', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), numerical_features)
    ]
)

X = data[categorical_features + numerical_features]
y = data['target']
groups = data['sentence_group']

# Define the SVM model with probability estimation enabled
model = SVC(
    kernel='linear',             # You can change to 'linear' or 'poly' if desired
    C=1.0,
    gamma = 0.1,
    class_weight='balanced',
    probability=True,         # Needed to get predict_proba for sentence-level scoring
    random_state=42
)

# Build pipeline
pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('classifier', model)
])

# Grouped CV using sentence-level accuracy
group_kfold = GroupKFold(n_splits=5)
accuracies = []

for fold_idx, (train_idx, test_idx) in enumerate(group_kfold.split(X, y, groups=groups)):
    print(f"Fold {fold_idx + 1}")
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    test_data = data.iloc[test_idx].copy()

    # Fit and predict
    pipeline.fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)[:, 1]

    # Attach predictions to test data
    test_data['proba'] = proba

    # Select one predicted root per sentence
    selected_roots = test_data.loc[test_data.groupby('sentence_id')['proba'].idxmax()]

    # Sentence-level accuracy
    accuracy = (selected_roots['target'] == 1).mean()
    accuracies.append(accuracy)
    print(f"Fold {fold_idx + 1} Sentence-Level Accuracy: {accuracy:.3f}")

# Final results
print("\n=== Cross-Validation Results ===")
print(f"Sentence-Level Accuracies: {accuracies}")
print(f"Mean Accuracy: {np.mean(accuracies):.3f} (±{np.std(accuracies):.3f})")
