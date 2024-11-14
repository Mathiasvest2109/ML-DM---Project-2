import numpy as np
import pandas as pd
from matplotlib.pylab import plt
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, roc_curve, roc_auc_score, confusion_matrix, precision_recall_curve, ConfusionMatrixDisplay
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

# Standardize features before applying Ridge and Logistic Regression
X_standardized = stats.zscore(X[:, 1:], axis=0)  # Standardize without the offset term
X_standardized = np.concatenate((np.ones((X_standardized.shape[0], 1)), X_standardized), axis=1)

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
for train_index, test_index in CV.split(X_standardized, y_regression):
    X_train = X_standardized[train_index]
    y_train = y_regression[train_index]
    X_test = X_standardized[test_index]
    y_test = y_regression[test_index]

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

# Find optimal lambda for Logistic Regression independently
Error_test_log_reg = np.empty(len(lambdas))
for i, lambda_val in enumerate(lambdas):
    mdl_log_reg = LogisticRegression(penalty="l2", C=1 / lambda_val, max_iter=1000, solver='liblinear')
    mdl_log_reg.fit(X_standardized, y_classification)
    Error_test_log_reg[i] = np.mean(mdl_log_reg.predict(X_standardized) != y_classification)  # Error rate

opt_lambda_log_reg_index = np.argmin(Error_test_log_reg)
opt_lambda_log_reg = lambdas[opt_lambda_log_reg_index]

# Train logistic regression with its own optimal lambda
mdl_log_reg_optimal = LogisticRegression(penalty="l2", C=1 / opt_lambda_log_reg, max_iter=1000, solver='liblinear')
mdl_log_reg_optimal.fit(X_standardized, y_classification)

# Extract standardized weights for logistic regression (using optimal logistic lambda)
weights_log_reg_optimal = mdl_log_reg_optimal.coef_[0]

# Display results for regularization parameters
print("\n--- Regularization Parameters ---")
print(f"Optimal lambda for Ridge Regression: {opt_lambda_rlr:.4e}")
print(f"Optimal lambda for Logistic Regression (independently found): {opt_lambda_log_reg:.4e}")

# Display standardized weights for feature comparison
print("\n--- Standardized Feature Comparison Between Ridge Regression and Logistic Regression ---")
print(f"{'Feature':<20} {'Ridge Weight':<15} {'Logistic Weight (Optimal)':<25}")
for attr, weight_rlr, weight_log_reg_opt in zip(attributeNames, optimal_weights_rlr, weights_log_reg_optimal):
    print(f"{attr:<20} {weight_rlr:<15.4f} {weight_log_reg_opt:<25.4f}")

# Plot the comparison of standardized weights between models
plt.figure(figsize=(12, 8))
index = np.arange(len(attributeNames))
bar_width = 0.35

# Plot standardized weights side-by-side
plt.bar(index, optimal_weights_rlr, bar_width, label="Ridge Regression (Regression)")
plt.bar(index + bar_width, weights_log_reg_optimal, bar_width, label="Logistic Regression (Classification)")

plt.xlabel("Features")
plt.ylabel("Standardized Weight Value")
plt.title("Standardized Feature Weights Comparison: Ridge vs Logistic Regression")
plt.xticks(index + bar_width / 2, attributeNames, rotation=90)
plt.legend()
plt.tight_layout()
plt.grid()
plt.show()

# Additional Visualization for Logistic Regression

# Generate predictions and probabilities for ROC and PR curves
y_pred_prob = mdl_log_reg_optimal.predict_proba(X_standardized)[:, 1]  # Probability for the positive class
y_pred = mdl_log_reg_optimal.predict(X_standardized)  # Predicted labels

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_classification, y_pred_prob)
roc_auc = roc_auc_score(y_classification, y_pred_prob)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Logistic Regression")
plt.legend()
plt.grid()
plt.show()

# Calculate Precision-Recall curve
precision, recall, pr_thresholds = precision_recall_curve(y_classification, y_pred_prob)

# Plot Precision-Recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label="Logistic Regression")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve for Logistic Regression")
plt.legend()
plt.grid()
plt.show()

# Plot Confusion Matrix
cm = confusion_matrix(y_classification, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=mdl_log_reg_optimal.classes_)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix for Logistic Regression")
plt.show()
