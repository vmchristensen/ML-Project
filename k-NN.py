
# Victoria – k-Nearest Neighbors Classifier

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import make_scorer, f1_score

# Load the preprocessed dataset
#data = pd.read_csv("preprocessed_train.csv")
# For degree + eccentricity only:
data = pd.read_csv("v3_preprocessed_train_degree_eccentricity.csv")

# Define features and target
categorical_features = ['language']
numerical_features = ['degree', 'eccentricity']  # adjust if needed

# Preprocessing: one-hot for language, scaling for numeric features
preprocessor = ColumnTransformer(
    transformers=[
        ('lang', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), numerical_features)  # important for distance-based models like k-NN
    ])

# Define model
model = KNeighborsClassifier(
    n_neighbors=5,       # Default, can be tuned
    weights='distance',  # Can also try 'uniform'
    metric='minkowski'   # Default (p=2 = Euclidean)
)

# Create pipeline (preprocessing + model)
pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('classifier', model)
])

# Prepare data
X = data[categorical_features + numerical_features]
y = data['target']
groups = data['sentence_id']

# F1 scorer
f1_scorer = make_scorer(f1_score, pos_label=1)

# Grouped CV
group_kfold = GroupKFold(n_splits=5)
cv_scores = cross_val_score(
    pipeline, X, y,
    cv=group_kfold,
    groups=groups,
    scoring=f1_scorer
)

# Output results
print(f"F1 Scores per fold: {cv_scores}")
print(f"Mean F1 (Class 1): {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")

# Fit final model (optional)
pipeline.fit(X, y)
