import importlib_resources
import numpy as np
import scipy.linalg as linalg
from matplotlib.pyplot import (
    cm,
    figure,
    imshow,
    legend,
    plot,
    show,
    subplot,
    title,
    xlabel,
    ylabel,
    yticks,
)
from scipy.io import loadmat
import pandas as pd
from ucimlrepo import fetch_ucirepo 

# fetch dataset 
heart_disease = fetch_ucirepo(id=45) 

# data (as pandas dataframes) 
X = heart_disease.data.features 
y = heart_disease.data.targets 

# Add the target variable 'num' as a column in X
X['num'] = y
print("Dataset with 'num' as target column:")
print(X)

# metadata 
# Uncomment if needed
# print(heart_disease.metadata) 

# variable information 
print("\nVariables Information:")
print(heart_disease.variables)

# List the feature names for X
print("\nFeatures (X):")
print(X.columns)

# Compute summary statistics
mean_x = X.mean()
std_x = X.std(ddof=1)
median_x = X.median()  # Use X.median() directly
range_x = X.max() - X.min()

# Display results
#print("\nSummary Statistics:")
#print("Mean:\n", mean_x)
#print("Standard Deviation:\n", std_x)
#print("Median:\n", median_x)
#print("Range:\n", range_x)

# Add preprocessing to account for missing values in dataset before introducing SVD
# For example, fill missing values with the mean of each column
X = X.fillna(X.mean())

# Print dataset after filling missing values (optional)
print("\nDataset after handling missing values:")
print(X)

# Proceed with SVD or other analysis here
