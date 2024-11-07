"""
Introduction to Machine Learning and Data Mining
Assignment 2
Task 3: Classification
 In this part of the report you are to solve a relevant classification problem for your data and statistically evaluate your result. The tasks will closely
 mirror what you just did in the last section. The three methods we will compare is a baseline, logistic regression, and one of the other four methods from below (referred
 to as method 2). What researches found in report 1 used was Random Forest and MLP (Multi-Layer Perceptron).
"""
import importlib_resources
import numpy as np
import pandas as pd
import sklearn.linear_model as lm
from matplotlib.pyplot import clim, figure, plot, show, subplot, title, xlabel, ylabel
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import model_selection
from ucimlrepo import fetch_ucirepo
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.neural_network import MLPClassifier
from dtuimldmtools import bmplot, feature_selector_lr, draw_neural_net, train_neural_net
import torch
from scipy import stats
from sklearn.datasets import fetch_openml
# optimization of lambda is done in exercise 8.1.2 and ANN in exercise 8.2.5

# Load the dataset from UCI repository
heart_disease = fetch_ucirepo(id=45)
data = heart_disease.data.features
targets =  heart_disease.data.targets 
print(targets.columns)
print(targets)
print(data.shape)
print(data.columns)
# variable information 
print(heart_disease.variables) 
# Drop rows with NaNs in X and update y accordingly
data_clean = data.dropna()
X = data_clean.to_numpy()
# target data is a ranking of heart disease from 0-4. 0 is no disease, 1-4 is disease present. 
# We want to do a binary classification, so we map 1-4 to 1 and 0 to 0.
y = targets.loc[data_clean.index, 'num'].astype(int).to_numpy()
y = (y > 0).astype(int)  # Map values 1-4 to 1 (disease present), and 0 to 0 (no disease)

#y = targets['num']  # Diagnosis of Heart Disease (num)(58)

# Normalize the data (z-score normalization)
X = stats.zscore(X)

# Baseline Model: Predict the most frequent class
most_frequent_class = np.bincount(y).argmax()
baseline_predictions = np.full(y.shape, most_frequent_class)
baseline_error_rate = np.mean(baseline_predictions != y)
print(f"Baseline error rate (most frequent class prediction): {baseline_error_rate:.4f}")

"""
# Apply one-of-K encoding to categorical variables (e.g., 'cp' for chest pain type)
X_encoded = pd.get_dummies(X, drop_first=True)

# Drop rows with missing values in both X_encoded and y
X_encoded = X_encoded.dropna()
y = y[X_encoded.index]  # Ensure y is aligned with X after dropping rows
# Get attribute names before converting to NumPy array
attributeNames = X_encoded.columns
X = X_encoded.to_numpy()
N, M = X.shape



# Standardize the training and set set based on training set mean and std
mu = np.mean(X_train, 0)
sigma = np.std(X_train, 0)

X_train = (X_train - mu) / sigma
X_test = (X_test - mu) / sigma
"""

# Parameters for cross-validation and complexity control
lambda_interval = np.logspace(-8, 2, 50)  # Regularization parameters for Logistic Regression
hidden_units_range = [1, 5, 10, 20, 50]   # Different numbers of hidden units for ANN
n_replicates = 2                          # Number of networks trained in each k-fold for ANN
max_iter = 10000                          # Maximum epochs for training ANN
K = 2                                     # Number of folds for cross-validation
testsize = 0.2
# Train-test split for Logistic Regression (without cross-validation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Initialize arrays to store results
train_error_rate_log_reg = np.zeros(len(lambda_interval))
test_error_rate_log_reg = np.zeros(len(lambda_interval))
coefficient_norm_log_reg = np.zeros(len(lambda_interval))

train_error_rate_ann = np.zeros(len(hidden_units_range))
test_error_rate_ann = np.zeros(len(hidden_units_range))


# Logistic Regression with different regularization strengths (no CV)
for k, lmbda in enumerate(lambda_interval):
    # Initialize Logistic Regression with the current lambda value
    mdl_log_reg = LogisticRegression(penalty="l2", C=1 / lmbda, max_iter=1000, solver='liblinear')
    mdl_log_reg.fit(X_train, y_train)
    
    # Make predictions on training and test sets
    y_train_est = mdl_log_reg.predict(X_train)
    y_test_est = mdl_log_reg.predict(X_test)
    
    # Calculate error rates
    train_error_rate_log_reg[k] = np.sum(y_train_est != y_train) / len(y_train)
    test_error_rate_log_reg[k] = np.sum(y_test_est != y_test) / len(y_test)
    
    # Calculate the L2 norm of the coefficient vector
    w_est = mdl_log_reg.coef_[0]
    coefficient_norm_log_reg[k] = np.sqrt(np.sum(w_est ** 2))


# Find the optimal lambda for Logistic Regression
min_error_log_reg = np.min(test_error_rate_log_reg)
opt_lambda_idx_log_reg = np.argmin(test_error_rate_log_reg)
opt_lambda_log_reg = lambda_interval[opt_lambda_idx_log_reg]

# ANN with cross-validation for different hidden units
CV = KFold(K, shuffle=True, random_state=42)

for k, hidden_units in enumerate(hidden_units_range):
    print(f"Training ANN with {hidden_units} hidden units")
    fold_errors = []
    for train_index, test_index in CV.split(X):
            # Extract training and test set for current CV fold, convert to tensors
        X_train = torch.Tensor(X[train_index, :])
        y_train = torch.Tensor(y[train_index]).reshape(-1, 1).float()  # Convert to float
        X_test = torch.Tensor(X[test_index, :])
        y_test = torch.Tensor(y[test_index]).reshape(-1, 1).float()    # Convert to float

         # Define the model for this configuration
        model = lambda: torch.nn.Sequential(
            torch.nn.Linear(X.shape[1], hidden_units),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_units, 1),
            torch.nn.Sigmoid()
        )
        
        # Train the neural network
        net, final_loss, _ = train_neural_net(model, torch.nn.BCELoss(), X=X_train, y=y_train, 
                                              n_replicates=n_replicates, max_iter=max_iter)
        
        # Test error for the fold
        y_sigmoid = net(X_test)
        y_test_est = (y_sigmoid > 0.5).type(dtype=torch.uint8)
        fold_errors.append((y_test_est != y_test).sum().item() / len(y_test))
    
    # Average test error for this hidden unit configuration
    test_error_rate_ann[k] = np.mean(fold_errors)
"""
# ANN with different hidden units
for k, hidden_units in enumerate(hidden_units_range):
    mdl_ann = MLPClassifier(hidden_layer_sizes=(hidden_units,), max_iter=1000, random_state=42)
    mdl_ann.fit(X_train, y_train)

    # Predictions and error rates
    y_train_est = mdl_ann.predict(X_train)
    y_test_est = mdl_ann.predict(X_test)

    train_error_rate_ann[k] = np.sum(y_train_est != y_train) / len(y_train)
    test_error_rate_ann[k] = np.sum(y_test_est != y_test) / len(y_test)

# Find the optimal number of hidden units for ANN
min_error_ann = np.min(test_error_rate_ann)
opt_hidden_units_idx_ann = np.argmin(test_error_rate_ann)
opt_hidden_units_ann = hidden_units_range[opt_hidden_units_idx_ann]
"""
# Find the optimal number of hidden units for ANN
min_error_ann = np.min(test_error_rate_ann)
opt_hidden_units_idx_ann = np.argmin(test_error_rate_ann)
opt_hidden_units_ann = hidden_units_range[opt_hidden_units_idx_ann]

print(f"Baseline error rate (most frequent class prediction): {baseline_error_rate:.4f}")
print(f"Optimal lambda for Logistic Regression: {opt_lambda_log_reg:.4e}")
print(f"Minimum test error (Logistic Regression): {min_error_log_reg:.4f}")
print(f"Optimal hidden units for ANN: {opt_hidden_units_ann}")
print(f"Minimum test error (ANN): {min_error_ann:.4f}")


# Plotting results for Logistic Regression
plt.figure(figsize=(10, 6))
plt.semilogx(lambda_interval, train_error_rate_log_reg * 100, label="Training error (Logistic Regression)")
plt.semilogx(lambda_interval, test_error_rate_log_reg * 100, label="Test error (Logistic Regression)")
plt.semilogx(opt_lambda_log_reg, min_error_log_reg * 100, 'o', label="Optimal λ")
plt.xlabel("Regularization strength, $\lambda$")
plt.ylabel("Error rate (%)")
plt.title("Logistic Regression: Error Rate vs Regularization")
plt.legend()
plt.grid()
plt.show()

# Plotting the coefficient norm vs regularization strength for Logistic Regression
plt.figure(figsize=(10, 6))
plt.semilogx(lambda_interval, coefficient_norm_log_reg, 'k')
plt.xlabel("Regularization strength, $\lambda$")
plt.ylabel("L2 Norm of Coefficients")
plt.title("Logistic Regression: Coefficient L2 Norm vs Regularization")
plt.grid()
plt.show()

# Plotting results for ANN
plt.figure(figsize=(10, 6))
plt.plot(hidden_units_range, test_error_rate_ann * 100, label="Test error (ANN)")
plt.plot(opt_hidden_units_ann, min_error_ann * 100, 'o', label="Optimal # Hidden Units")
plt.xlabel("Number of Hidden Units")
plt.ylabel("Test Error Rate (%)")
plt.title("ANN: Error Rate vs Number of Hidden Units")
plt.legend()
plt.grid()
plt.show()

print("Used Exercise 8.1.2 and 8.2.5")
