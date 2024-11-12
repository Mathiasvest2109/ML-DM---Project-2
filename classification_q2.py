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
from sklearn.metrics import accuracy_score
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
max_iter = 1000                          # Maximum epochs for training ANN
K = 2                                     # Number of folds for cross-validation
# Define the number of outer and inner folds
outer_folds = 5
inner_folds = 5
outer_cv = KFold(n_splits=outer_folds, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=inner_folds, shuffle=True, random_state=42)

# Initialize variables 
Error_train = np.empty((outer_folds, 1))
Error_test = np.empty((outer_folds, 1))
Error_baseline = np.empty((outer_folds, 1))
Error_train_log_reg = np.empty((outer_folds, 1))
Error_test_log_reg = np.empty((outer_folds, 1))
Error_train_ann = np.empty((outer_folds, 1))
Error_test_ann = np.empty((outer_folds, 1))

# Add these lists to store the values for each fold
best_log_reg_lambdas = []      # Store best lambda for each outer fold in Logistic Regression
best_ann_hidden_units = []      # Store best hidden units for each outer fold in ANN
log_reg_test_errors = []        # Store test errors for Logistic Regression in each outer fold
ann_test_errors = []            # Store test errors for ANN in each outer fold
baseline_test_errors = []       # Store baseline errors in each outer fold

#testsize = 0.2
# Train-test split for Logistic Regression (without cross-validation)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Initialize arrays to store results
train_error_rate_log_reg = np.zeros(len(lambda_interval))
test_error_rate_log_reg = np.zeros(len(lambda_interval))
coefficient_norm_log_reg = np.zeros(len(lambda_interval))

train_error_rate_ann = np.zeros(len(hidden_units_range))
test_error_rate_ann = np.zeros(len(hidden_units_range))
# Outer Cross-Validation Loop
for k, (train_index, test_index) in enumerate(outer_cv.split(X, y)):
    print(f'Outer fold {k+1}/{outer_folds}')
    # Extract training and test sets for current outer fold
    X_train_outer, X_test_outer = X[train_index], X[test_index]
    y_train_outer, y_test_outer = y[train_index], y[test_index]
    
    # Baseline model (predict most frequent class in training set)
    most_frequent_class = np.bincount(y_train_outer).argmax()
    baseline_predictions = np.full(y_test_outer.shape, most_frequent_class)
    baseline_error = np.mean(baseline_predictions != y_test_outer)
    Error_baseline[k] = baseline_error
    baseline_test_errors.append(baseline_error * 100)  # Store baseline error in percentage


    # Inner cross-validation for Logistic Regression to find best lambda
    best_log_reg_error = float('inf')
    best_lambda = None
    average_errors_lambda = []
    for lmbda in lambda_interval:
        inner_errors = []
        for j, (inner_train_idx, inner_val_idx) in enumerate(inner_cv.split(X_train_outer, y_train_outer)):
            print(f'Inner fold {j+1}/{inner_folds}, lambda: {lmbda}', "Outer fold:", k+1)
        #for inner_train_idx, inner_val_idx in inner_cv.split(X_train_outer, y_train_outer):
        #    print(f'Inner fold {k+1}/{outer_folds}, lambda: {lmbda}')
            X_train_inner, X_val_inner = X[inner_train_idx], X[inner_val_idx]
            y_train_inner, y_val_inner = y[inner_train_idx], y[inner_val_idx]
            
            # Train Logistic Regression
            log_reg_model = LogisticRegression(penalty="l2", C=1 / lmbda, max_iter=1000, solver='liblinear')
            log_reg_model.fit(X_train_inner, y_train_inner)
            y_val_pred = log_reg_model.predict(X_val_inner)
            inner_errors.append(1 - accuracy_score(y_val_inner, y_val_pred))
        
        avg_inner_error = np.mean(inner_errors)
        if avg_inner_error < best_log_reg_error:
            best_log_reg_error = avg_inner_error
            best_lambda = lmbda
        print('Best lambda:', best_lambda)
        average_errors_lambda.append(avg_inner_error)
    
    # Evaluate Logistic Regression on outer test set
    log_reg_model = LogisticRegression(penalty="l2", C=1 / best_lambda, max_iter=1000, solver='liblinear')
    log_reg_model.fit(X_train_outer, y_train_outer)
    y_test_pred = log_reg_model.predict(X_test_outer)
    Error_test_log_reg[k] = 1 - accuracy_score(y_test_outer, y_test_pred)
    log_reg_test_errors.append(Error_test_log_reg[k][0] * 100)  # Store test error in percentage for this fold
    best_log_reg_lambdas.append(best_lambda)  # Store best lambda for this fold

    # Inner cross-validation for ANN (find best hidden units)
    best_ann_error = float('inf')
    best_hidden_units = None
    average_errors_hidden = []

    for hidden_units in hidden_units_range:
        print(f'Inner fold {k+1}/{outer_folds}, hidden units: {hidden_units}', "Outer fold:", k+1)
        print('Outer fold:', k+1)
        inner_errors = []
        for j, (inner_train_idx, inner_val_idx) in enumerate(inner_cv.split(X_train_outer, y_train_outer)):  # Use `j` for inner folds
            print(f'  Inner fold {j+1}/{inner_folds}, hidden units: {hidden_units}')
        #for inner_train_idx, inner_val_idx in inner_cv.split(X_train_outer, y_train_outer):
        #    print(f'Inner fold {k+1}/{outer_folds}, hidden units: {hidden_units}')
        #    print(f'Outer fold {k+1}/{outer_folds}')

            # Convert to torch tensors
            X_train_inner = torch.Tensor(X[inner_train_idx])
            y_train_inner = torch.Tensor(y[inner_train_idx]).reshape(-1, 1).float()
            X_val_inner = torch.Tensor(X[inner_val_idx])
            y_val_inner = torch.Tensor(y[inner_val_idx]).reshape(-1, 1).float()
            
            # Define and train ANN model
            model = lambda: torch.nn.Sequential(
                torch.nn.Linear(X.shape[1], hidden_units),
                torch.nn.Tanh(),
                torch.nn.Linear(hidden_units, 1)
            )
            net, final_loss, _ = train_neural_net(model, torch.nn.BCEWithLogitsLoss(), X=X_train_inner, y=y_train_inner,
                                                  n_replicates=n_replicates, max_iter=max_iter)
            
            # Validate and calculate error rate
            y_val_pred = (torch.sigmoid(net(X_val_inner)) > 0.5).type(torch.uint8)
            error_rate = (y_val_pred != y_val_inner).sum().item() / len(y_val_inner)
            inner_errors.append(error_rate)
        
        avg_inner_error = np.mean(inner_errors)
        if avg_inner_error < best_ann_error:
            best_ann_error = avg_inner_error
            best_hidden_units = hidden_units
        print('Best hidden units:', best_hidden_units)
        average_errors_hidden.append(avg_inner_error)
    
    # Evaluate ANN on outer test set
    X_train_outer_torch = torch.Tensor(X_train_outer)
    y_train_outer_torch = torch.Tensor(y_train_outer).reshape(-1, 1).float()
    net, final_loss, _ = train_neural_net(lambda: torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], best_hidden_units),
        torch.nn.Tanh(),
        torch.nn.Linear(best_hidden_units, 1)
    ), torch.nn.BCEWithLogitsLoss(), X=X_train_outer_torch, y=y_train_outer_torch, 
       n_replicates=n_replicates, max_iter=max_iter)
    X_test_outer_torch = torch.Tensor(X_test_outer)
    y_test_outer_torch = torch.Tensor(y_test_outer).reshape(-1, 1).float()
    y_test_pred = (torch.sigmoid(net(X_test_outer_torch)) > 0.5).type(torch.uint8)
    Error_test_ann[k] = (y_test_pred != y_test_outer_torch).sum().item() / len(y_test_outer_torch)
    ann_test_error = (y_test_pred != y_test_outer_torch).sum().item() / len(y_test_outer_torch)
    
    ann_test_errors.append(ann_test_error * 100)  # Store ANN error in percentage for this fold
    best_ann_hidden_units.append(best_hidden_units)  # Store best hidden units for this fold

    print(f'Fold {k+1}/{outer_folds}')
    #print(f'Baseline Error old: {baseline_error:.4f}, LogReg Error: {Error_test_log_reg[k]:.4f}, ANN Error: {Error_test_ann[k]:.4f}')
    print(f'Baseline Error: {baseline_error:.4f}, LogReg Error: {Error_test_log_reg[k][0]:.4f}, ANN Error: {ann_test_error:.4f}')

results_table = pd.DataFrame({
    "Outer fold": np.arange(1, outer_folds + 1),
    "Method 2 (Hidden Units)": best_ann_hidden_units,
    "Method 2 Error (%)": ann_test_errors,
    "Logistic Regression (Lambda)": best_log_reg_lambdas,
    "Logistic Regression Error (%)": log_reg_test_errors,
    "Baseline Error (%)": baseline_test_errors
})
print(results_table)

# Plotting
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Plot for the number of hidden units
ax[0].plot(hidden_units_range, average_errors_hidden, 'o-')
ax[0].set_xlabel("Number of hidden units")
ax[0].set_ylabel("Generalization error")
ax[0].set_title("Generalization error vs. number of hidden units")
ax[0].grid()

# Plot for lambda
ax[1].plot(lambda_interval, average_errors_lambda, 'o-')
ax[1].set_xscale("log")  # Log scale for lambda
ax[1].set_xlabel("Regularization parameter (lambda)")
ax[1].set_ylabel("Generalization error")
ax[1].set_title("Generalization error vs. regularization parameter (lambda)")
ax[1].grid()

plt.tight_layout()
plt.show()
"""

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
plt.plot(hidden_units_range, train_error_rate_ann * 100, label="Training error (ANN)")
plt.plot(hidden_units_range, test_error_rate_ann * 100, label="Test error (ANN)")
plt.plot(opt_hidden_units_ann, min_error_ann * 100, 'o', label="Optimal # Hidden Units")
plt.xlabel("Number of Hidden Units")
plt.ylabel("Test Error Rate (%)")
plt.title("ANN: Error Rate vs Number of Hidden Units")
plt.legend()
plt.grid()
plt.show()
"""
print("Used Exercise 8.1.2 and 8.2.5")
