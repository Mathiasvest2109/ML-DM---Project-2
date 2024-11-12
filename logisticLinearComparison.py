import numpy as np
import pandas as pd
from matplotlib.pylab import figure, grid, legend, semilogx, show, xlabel, ylabel, title
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error
from ucimlrepo import fetch_ucirepo
from scipy import stats

# Fetch the heart disease dataset
heart_disease = fetch_ucirepo(id=45)
data = heart_disease.data.features

# Set the target variable for regression as 'thalach' (Max Heart Rate)
y_regression = data['thalach'].to_numpy()

# Set the target variable for classification (1 if disease present, 0 otherwise)
targets = heart_disease.data.targets
y_classification = (targets['num'] > 0).astype(int).to_numpy()

# Select features for the models and remove the target variable 'thalach' from X
X = data.drop(columns=['thalach'])

# Apply one-of-K encoding to categorical variables (e.g., 'cp' for chest pain type)
X_encoded = pd.get_dummies(X, drop_first=True)

# Drop rows with missing values in both X_encoded and y
X_encoded = X_encoded.dropna()
y_regression = y_regression[X_encoded.index]  # Align y for regression
y_classification = y_classification[X_encoded.index]  # Align y for classification

# Convert to numpy arrays
X = X_encoded.to_numpy()
attributeNames = X_encoded.columns.tolist()
N, M = X.shape

# Add an offset attribute (bias term) to X
X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
attributeNames = ["Offset"] + attributeNames
M += 1  # Update the number of features

# Cross-validation setup for ridge regression (regression model)
K = 10  # Number of cross-validation folds
CV = KFold(n_splits=K, shuffle=True, random_state=0)
lambdas = np.logspace(-3, 2, 20)

# Initialize variables to store errors and weights for regression
Error_train_rlr = np.empty((K, len(lambdas)))
Error_test_rlr = np.empty((K, len(lambdas)))
w_rlr = np.empty((M, K, len(lambdas)))

# Cross-validation loop for ridge regression
k = 0
for train_index, test_index in CV.split(X, y_regression):
    X_train = X[train_index]
    y_train = y_regression[train_index]
    X_test = X[test_index]
    y_test = y_regression[test_index]

    # Standardize (z-score) based on training set
    mu = np.mean(X_train[:, 1:], axis=0)
    sigma = np.std(X_train[:, 1:], axis=0)
    X_train[:, 1:] = (X_train[:, 1:] - mu) / sigma
    X_test[:, 1:] = (X_test[:, 1:] - mu) / sigma

    # Loop over lambdas
    for i, lambda_val in enumerate(lambdas):
        ridge = Ridge(alpha=lambda_val, fit_intercept=False)
        ridge.fit(X_train, y_train)
        
        w_rlr[:, k, i] = ridge.coef_
        Error_train_rlr[k, i] = mean_squared_error(y_train, ridge.predict(X_train))
        Error_test_rlr[k, i] = mean_squared_error(y_test, ridge.predict(X_test))

    k += 1

# Mean errors and optimal lambda for ridge regression
mean_test_error_rlr = Error_test_rlr.mean(axis=0)
opt_lambda_rlr_index = np.argmin(mean_test_error_rlr)
opt_lambda_rlr = lambdas[opt_lambda_rlr_index]
optimal_weights_rlr = w_rlr[:, :, opt_lambda_rlr_index].mean(axis=1)  # Average weights across folds

# Version 1: Use optimal lambda from Ridge Regression in Logistic Regression
X_train, X_test, y_train, y_test = train_test_split(X, y_classification, test_size=0.2, stratify=y_classification, random_state=42)
mdl_log_reg_from_ridge = LogisticRegression(penalty="l2", C=1 / opt_lambda_rlr, max_iter=1000, solver='liblinear')
mdl_log_reg_from_ridge.fit(X_train, y_train)

# Extract weights for logistic regression (using Ridge lambda)
weights_log_reg_from_ridge = mdl_log_reg_from_ridge.coef_[0]

# Version 2: Find optimal lambda for Logistic Regression independently
Error_test_log_reg = np.empty(len(lambdas))
for i, lambda_val in enumerate(lambdas):
    mdl_log_reg = LogisticRegression(penalty="l2", C=1 / lambda_val, max_iter=1000, solver='liblinear')
    mdl_log_reg.fit(X_train, y_train)
    Error_test_log_reg[i] = np.mean(mdl_log_reg.predict(X_test) != y_test)  # Error rate

opt_lambda_log_reg_index = np.argmin(Error_test_log_reg)
opt_lambda_log_reg = lambdas[opt_lambda_log_reg_index]

# Train logistic regression with its own optimal lambda
mdl_log_reg_optimal = LogisticRegression(penalty="l2", C=1 / opt_lambda_log_reg, max_iter=1000, solver='liblinear')
mdl_log_reg_optimal.fit(X_train, y_train)

# Extract weights for logistic regression (using optimal logistic lambda)
weights_log_reg_optimal = mdl_log_reg_optimal.coef_[0]

# Display results for regularization parameters
print("\n--- Regularization Parameters ---")
print(f"Optimal lambda for Ridge Regression (used in both models in Version 1): {opt_lambda_rlr:.4e}")
print(f"Optimal lambda for Logistic Regression (independently found in Version 2): {opt_lambda_log_reg:.4e}")

# Display weights for feature comparison
print("\n--- Feature Comparison Between Ridge Regression and Logistic Regression ---")
print(f"{'Feature':<20} {'Ridge Weight':<15} {'Logistic Weight (Version 1)':<25} {'Logistic Weight (Version 2)':<25}")
for attr, weight_rlr, weight_log_reg_ridge, weight_log_reg_opt in zip(attributeNames, optimal_weights_rlr, weights_log_reg_from_ridge, weights_log_reg_optimal):
    print(f"{attr:<20} {weight_rlr:<15.4f} {weight_log_reg_ridge:<25.4f} {weight_log_reg_opt:<25.4f}")

# Plot the comparison of weights between models
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
index = np.arange(len(attributeNames))
bar_width = 0.25

# Plot weights side-by-side
plt.bar(index, optimal_weights_rlr, bar_width, label="Ridge Regression (Regression)")
plt.bar(index + bar_width, weights_log_reg_from_ridge, bar_width, label="Logistic Regression (Using Ridge Lambda - Version 1)")
plt.bar(index + 2 * bar_width, weights_log_reg_optimal, bar_width, label="Logistic Regression (Own Optimal Lambda - Version 2)")

plt.xlabel("Features")
plt.ylabel("Weight Value")
plt.title("Feature Weights Comparison: Ridge vs Logistic Regression")
plt.xticks(index + bar_width, attributeNames, rotation=90)
plt.legend()
plt.tight_layout()
plt.grid()
plt.show()
