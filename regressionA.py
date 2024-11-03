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

X['target'] = y
print(X)
  
# metadata 
#print(heart_disease.metadata) 
  
# variable information 
print(heart_disease.variables) 

# List the feature names for X
print("Features (X):")
print(X.columns)

# List the target column name for y
print("\nTarget (y):")
print(y.columns)

# Compute summary statistics
mean_x = X.mean()
std_x = X.std(ddof=1)
median_x = np.median(X)
range_x = X.max() - X.min()

# Display results
print("Vector:", X)
print("Mean:", mean_x)
print("Standard Deviation:", std_x)
print("Median:", median_x)
print("Range:", range_x)

# Save the features dataset (X) to a CSV file
X.to_csv("features.csv", index=False)

#Add prprocessing to account for missing values in dataset before introducing SVD
