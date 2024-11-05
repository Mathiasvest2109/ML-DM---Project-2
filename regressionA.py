import numpy as np
import pandas as pd
import sklearn.linear_model as lm
from matplotlib.pylab import (
    figure,
    grid,
    legend,
    semilogx,
    show,
    subplot,
    title,
    xlabel,
    ylabel,
)
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from ucimlrepo import fetch_ucirepo

# Fetch the heart disease dataset
heart_disease = fetch_ucirepo(id=45)
data = heart_disease.data.features

# Set the target variable as 'thalach' (Max Heart Rate)
y = data['thalach'].to_numpy()

# Select features for the regression model and remove the target variable 'thalach' from X
X = data.drop(columns=['thalach'])

# Apply one-of-K encoding to categorical variables (e.g., 'cp' for chest pain type)
X_encoded = pd.get_dummies(X, drop_first=True)

# Drop rows with missing values in both X_encoded and y
X_encoded = X_encoded.dropna()
y = y[X_encoded.index]  # Ensure y is aligned with X after dropping rows

# Convert to numpy arrays for efficient calculations
X = X_encoded.to_numpy()
attributeNames = X_encoded.columns.tolist()
N, M = X.shape

# Add an offset attribute (bias term) to X
X = np.concatenate((np.ones((X.shape[0], 1)), X), 1)
attributeNames = ["Offset"] + attributeNames
M += 1  # Update the number of features

# Cross-validation setup
K = 10  # Number of cross-validation folds
CV = KFold(n_splits=K, shuffle=True, random_state=0)

# Values of lambda for regularization
lambdas = np.logspace(-3, 2, 20)

# Initialize variables to store errors and weights
Error_train = np.empty((K, 1))
Error_test = np.empty((K, 1))
Error_train_rlr = np.empty((K, len(lambdas)))
Error_test_rlr = np.empty((K, len(lambdas)))
w_rlr = np.empty((M, K, len(lambdas)))

# Cross-validation loop
k = 0
for train_index, test_index in CV.split(X, y):
    # Extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]

    # Standardize based on training set and save mean & std for use on test set
    mu = np.mean(X_train[:, 1:], axis=0)
    sigma = np.std(X_train[:, 1:], axis=0)

    # Standardize training and test sets (excluding the offset/bias term)
    X_train[:, 1:] = (X_train[:, 1:] - mu) / sigma
    X_test[:, 1:] = (X_test[:, 1:] - mu) / sigma

    # Compute mean squared error without using any features (baseline)
    Error_train[k] = mean_squared_error(y_train, np.repeat(y_train.mean(), len(y_train)))
    Error_test[k] = mean_squared_error(y_test, np.repeat(y_train.mean(), len(y_test)))

    # Loop over each lambda and train a ridge regression model
    for i, lambda_val in enumerate(lambdas):
        # Train Ridge Regression with current lambda
        ridge = Ridge(alpha=lambda_val, fit_intercept=False)
        ridge.fit(X_train, y_train)
        
        # Store weights
        w_rlr[:, k, i] = ridge.coef_

        # Compute training and test error
        Error_train_rlr[k, i] = mean_squared_error(y_train, ridge.predict(X_train))
        Error_test_rlr[k, i] = mean_squared_error(y_test, ridge.predict(X_test))

    k += 1

# Calculate mean errors across folds for each lambda
mean_train_error = Error_train_rlr.mean(axis=0)
mean_test_error = Error_test_rlr.mean(axis=0)

# Find the optimal lambda based on the lowest mean test error
opt_lambda_index = np.argmin(mean_test_error)
opt_lambda = lambdas[opt_lambda_index]

# Plotting mean squared error vs. lambda for visualization
figure(figsize=(8, 6))
semilogx(lambdas, mean_train_error, 'b.-', label="Mean Training Error")
semilogx(lambdas, mean_test_error, 'r.-', label="Mean Test Error (Generalization Error)")
xlabel("Regularization Parameter (Lambda)")
ylabel("Mean Squared Error")
title("Effect of Lambda on Generalization Error in Least Squares Regression")
legend()
grid()
show()

# Display results
print("Best lambda:", opt_lambda)
print("Mean training error with best lambda:", mean_train_error[opt_lambda_index])
print("Mean test error with best lambda:", mean_test_error[opt_lambda_index])

print("Weights with optimal lambda:")
for m in range(M):
    print(f"{attributeNames[m]:>15} {np.round(w_rlr[m, :, opt_lambda_index].mean(), 2):>15}")