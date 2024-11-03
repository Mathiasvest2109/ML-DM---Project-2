import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo

# Load the dataset from UCI repository
heart_disease = fetch_ucirepo(id=45)
data = heart_disease.data.features

# Set the target variable as 'thalach' (Max Heart Rate)
y = data['thalach']

# Select features for the regression model and remove the target variable 'thalach' from X
X = data.drop(columns=['thalach'])

# Apply one-of-K encoding to categorical variables (e.g., 'cp' for chest pain type)
X_encoded = pd.get_dummies(X, drop_first=True)

# Drop rows with missing values in both X_encoded and y
X_encoded = X_encoded.dropna()
y = y[X_encoded.index]  # Ensure y is aligned with X after dropping rows

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=0)

# Standardize the features in both training and testing sets
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Fit and transform on training data
X_test = scaler.transform(X_test)        # Only transform on testing data

# Train a linear regression model on the training data
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Model evaluation metrics on testing data
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print("\nModel Evaluation Metrics (on testing data):")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")

# Display the intercept and coefficients of the model
print("\nModel Intercept:", model.intercept_)
print("Model Coefficients:", model.coef_)

# Plot actual vs predicted values on testing data
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Max Heart Rate (Actual)")
plt.ylabel("Max Heart Rate (Predicted)")
plt.title("Predicted vs Actual Max Heart Rate (thalach) on Testing Data")
plt.legend(["Predicted vs Actual"])
plt.show()
