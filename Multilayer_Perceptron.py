import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset
import time
from collections import defaultdict
import random

# Configuration
RANDOM_SEED = 42
NUM_HYPERPARAM_TRIALS = 20
torch.manual_seed(RANDOM_SEED)

# Load and preprocess data
print("Loading and preprocessing data...")
data = pd.read_csv("preprocessed_train.csv")

categorical_features = ['language']
numerical_features = ['degree', 'closeness', 'betweenness', 'pagerank', 'harmonic', 'eigenvector', 'eccentricity']

preprocessor = ColumnTransformer(
    transformers=[
        ('lang', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), numerical_features)
    ],
    sparse_threshold=0
)

X = preprocessor.fit_transform(data)
y = data['target'].values
groups = data['sentence_id'].values

# Convert to PyTorch tensors
X_tensor = torch.FloatTensor(X)
y_tensor = torch.LongTensor(y)

# Define MLP model
class CustomMLP(nn.Module):
    def __init__(self, input_size, hidden_layers, dropout_rate):
        super().__init__()
        layers = []
        in_features = input_size
        
        for out_features in hidden_layers:
            layers += [
                nn.Linear(in_features, out_features),
                nn.ReLU(),
                nn.Dropout(p=dropout_rate)
            ]
            in_features = out_features
            
        layers.append(nn.Linear(in_features, 2))
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)

# Hyperparameter search space
param_grid = {
    'learning_rate': [0.001, 0.0005, 0.0001],
    'hidden_layers': [(128, 64), (64, 32), (256, 128, 64)],
    'dropout_rate': [0.2, 0.3, 0.4],
    'batch_size': [128, 256, 512],
    'weight_decay': [0, 1e-4, 1e-5],
    'optimizer': ['adam', 'sgd']
}

# Generate random parameter combinations
print(f"\nGenerating {NUM_HYPERPARAM_TRIALS} hyperparameter combinations...")
param_combinations = [
    {
        'learning_rate': random.choice(param_grid['learning_rate']),
        'hidden_layers': random.choice(param_grid['hidden_layers']),
        'dropout_rate': random.choice(param_grid['dropout_rate']),
        'batch_size': random.choice(param_grid['batch_size']),
        'weight_decay': random.choice(param_grid['weight_decay']),
        'optimizer': random.choice(param_grid['optimizer'])
    }
    for _ in range(NUM_HYPERPARAM_TRIALS)
]

# Hyperparameter search setup
best_score = -np.inf
best_params = None
search_history = defaultdict(list)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}\n")

# Main hyperparameter search loop
for trial_idx, params in enumerate(param_combinations):
    trial_start = time.time()
    print(f"\n=== Trial {trial_idx+1}/{NUM_HYPERPARAM_TRIALS} ===")
    print("Parameters:", params)
    
    fold_accuracies = []
    group_kfold = GroupKFold(n_splits=5)
    
    # Cross-validation loop
    for fold_idx, (train_idx, test_idx) in enumerate(group_kfold.split(X, y, groups=groups)):
        fold_start = time.time()
        print(f"\n  Fold {fold_idx+1}/5 -", end=" ")
        
        # Split data
        X_train, X_test = X_tensor[train_idx], X_tensor[test_idx]
        y_train, y_test = y_tensor[train_idx], y_tensor[test_idx]
        
        # Class weights
        classes = np.unique(y_train.numpy())
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train.numpy())
        class_weights = torch.FloatTensor(class_weights).to(device)
        
        # Model initialization
        model = CustomMLP(
            input_size=X.shape[1],
            hidden_layers=params['hidden_layers'],
            dropout_rate=params['dropout_rate']
        ).to(device)
        
        # Optimizer setup
        if params['optimizer'] == 'adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=params['learning_rate'],
                weight_decay=params['weight_decay']
            )
        else:
            optimizer = optim.SGD(
                model.parameters(),
                lr=params['learning_rate'],
                momentum=0.9,
                weight_decay=params['weight_decay']
            )
        
        # Training setup
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        train_dataset = TensorDataset(X_train.to(device), y_train.to(device))
        train_loader = DataLoader(
            train_dataset,
            batch_size=params['batch_size'],
            shuffle=True
        )
        
        # Early stopping
        best_val_loss = np.inf
        patience = 5
        patience_counter = 0
        
        # Training loop
        for epoch in range(100):
            model.train()
            epoch_loss = 0
            
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            # Validation check
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_test.to(device))
                val_loss = criterion(val_outputs, y_test.to(device)).item()
            
            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}", end=" ")
                break
        
        # Calculate accuracy
        with torch.no_grad():
            probabilities = torch.softmax(model(X_test.to(device)), dim=1)[:, 1].cpu().numpy()
        
        test_data = data.iloc[test_idx].copy()
        test_data['proba'] = probabilities
        selected_roots = test_data.loc[test_data.groupby('sentence_id')['proba'].idxmax()]
        accuracy = (selected_roots['target'] == 1).mean()
        fold_accuracies.append(accuracy)
        
        fold_time = time.time() - fold_start
        print(f"Accuracy: {accuracy:.3f} | Time: {fold_time:.1f}s")

    # Trial evaluation
    trial_score = np.mean(fold_accuracies)
    search_history['params'].append(params)
    search_history['scores'].append(trial_score)
    
    if trial_score > best_score:
        best_score = trial_score
        best_params = params
        print(f"\n🔥 New best score: {best_score:.3f}!")
    
    trial_time = time.time() - trial_start
    print(f"\nTrial completed in {trial_time:.1f}s")
    print(f"Current best accuracy: {best_score:.3f}")

# Final results
print("\n=== HYPERPARAMETER SEARCH RESULTS ===")
print(f"Best validation accuracy: {best_score:.3f}")
print("Best parameters:")
for key, value in best_params.items():
    print(f"- {key}: {value}")

"""
🔥 New best score: 0.352!

Trial completed in 177.9s
Current best accuracy: 0.352

=== HYPERPARAMETER SEARCH RESULTS ===
Best validation accuracy: 0.352
Best parameters:
- learning_rate: 0.001
- hidden_layers: (128, 64)
- dropout_rate: 0.4
- batch_size: 512
- weight_decay: 1e-05
- optimizer: adam
"""