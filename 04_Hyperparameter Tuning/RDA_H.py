import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupKFold
from pathlib import Path


class RDA(BaseEstimator, ClassifierMixin):
    """
    Custom Regularized Discriminant Analysis (RDA) Classifier.
    - alpha: Interpolates between LDA (0) and QDA (1)
    - gamma: Regularizes covariance matrices (0: no shrinkage, 1: diagonal)
    """
    def __init__(self, alpha=0.5, gamma=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.classes_ = None
        self.priors_ = None
        self.means_ = None
        self.covariances_ = None
        self.pooled_cov_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        self.priors_ = np.array([np.mean(y == c) for c in self.classes_])
        self.means_ = np.array([X[y == c].mean(axis=0) for c in self.classes_])
        
        self.covariances_ = []
        for c in self.classes_:
            X_c = X[y == c]
            cov = np.cov(X_c, rowvar=False)
            shrunk_cov = (1 - self.gamma) * cov + self.gamma * np.diag(np.diag(cov))
            self.covariances_.append(shrunk_cov)
        
        self.pooled_cov_ = np.sum([self.priors_[i] * self.covariances_[i] 
                                 for i in range(n_classes)], axis=0)
        return self

    def predict_proba(self, X):
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        proba = np.zeros((n_samples, n_classes))
        
        for i, c in enumerate(self.classes_):
            diff = X - self.means_[i]
            cov = (1 - self.alpha) * self.pooled_cov_ + self.alpha * self.covariances_[i]
            cov_inv = np.linalg.pinv(cov)
            log_det = np.linalg.slogdet(cov)[1]
            
            dist = np.sum(diff @ cov_inv * diff, axis=1)
            proba[:, i] = -0.5 * (log_det + dist) + np.log(self.priors_[i])
        
        proba = np.exp(proba - proba.max(axis=1, keepdims=True))
        proba /= proba.sum(axis=1, keepdims=True)
        return proba

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

# Load data
script_dir = Path(__file__).parent
data_path = script_dir.parent / "01_Data" / "preprocessed_train.csv"
data = pd.read_csv(data_path)
categorical_features = ['language']
numerical_features = ['degree', 'closeness', 'betweenness', 'pagerank', 'harmonic', 'eigenvector', 'eccentricity']

preprocessor = ColumnTransformer(
    transformers=[
        ('lang', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        ('num', 'passthrough', numerical_features)
    ])

X = preprocessor.fit_transform(data)
y = data['target']
groups = data['sentence_group']

# Hyperparameter grid
param_grid = {
    'alpha': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'gamma': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
}

# Tuning setup
group_kfold = GroupKFold(n_splits=5)
best_score = 0
best_params = {}

# Grid search
for alpha in param_grid['alpha']:
    for gamma in param_grid['gamma']:
        print(f"\n=== Testing alpha={alpha}, gamma={gamma} ===")
        fold_accuracies = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(group_kfold.split(X, y, groups=groups)):
            # Data splitting
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Model training
            model = RDA(alpha=alpha, gamma=gamma)
            model.fit(X_train, y_train)
            
            # Prediction and grouping
            test_data = data.iloc[test_idx].copy()
            test_data['proba'] = model.predict_proba(X_test)[:, 1]
            selected_roots = test_data.loc[test_data.groupby('sentence_id')['proba'].idxmax()]
            
            # Accuracy calculation
            fold_accuracy = (selected_roots['target'] == 1).mean()
            fold_accuracies.append(fold_accuracy)
            print(f"Fold {fold_idx+1} accuracy: {fold_accuracy:.3f}")
        
        # Mean accuracy for current parameters
        mean_accuracy = np.mean(fold_accuracies)
        print(f"Mean accuracy for alpha={alpha}, gamma={gamma}: {mean_accuracy:.3f}")
        
        # Update best parameters
        if mean_accuracy > best_score:
            best_score = mean_accuracy
            best_params = {'alpha': alpha, 'gamma': gamma}

# Results
print("\n=== Final Results ===")
print(f"Best parameters: {best_params}")
print(f"Best mean accuracy: {best_score:.3f}")


"""
=== Final Results ===
Best parameters: {'alpha': 1.0, 'gamma': 0.6}
Best mean accuracy: 0.305
"""