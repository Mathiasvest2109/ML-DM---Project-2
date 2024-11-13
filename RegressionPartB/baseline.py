import numpy as np
from sklearn import model_selection
from ucimlrepo import fetch_ucirepo
import pandas as pd
import inspect
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

import torch
from scipy import stats

from dtuimldmtools import draw_neural_net, train_neural_net

# Load the dataset from UCI repository
heart_disease = fetch_ucirepo(id=45)
X = heart_disease.data.features
X['target'] = heart_disease.data.targets
X['target'] = pd.cut(X['target'], [-1,0, np.inf],labels=['0.00000000001', '1'])

y=X['thalach']
X = X.drop(columns=['thalach'])
# Apply one-of-K encoding to categorical variables (e.g., 'cp' for chest pain type)
X_encoded = pd.get_dummies(X, drop_first=True)

# Drop rows with missing values in both X_encoded and y
X_encoded = X_encoded.dropna()
y = y[X_encoded.index]  # Ensure y is aligned with X after dropping rows

X = X_encoded.to_numpy(dtype=np.float64)
y = y.to_numpy()
y= np.reshape(y,(-1,1))
attributeNames = X_encoded.columns.tolist()
N, M = X.shape
C = 2

k1 = 10

CV1 = model_selection.KFold(k1, shuffle=True)
for i, (D_par_i, D_test_i) in enumerate(CV1.split(X, y)):
    
    #print("fold",i,"errors",errors)
    X_par = (X[D_par_i, :])
    y_par = (y[D_par_i])
    X_test = (X[D_test_i, :])
    y_test = (y[D_test_i])
    
    baseline = np.repeat(y_par.mean(), len(y_test))
        
    mse = mean_squared_error(y_test, baseline)

    print("\nOuter fold:",i,"\tEtest_{0}".format(i),mse)
        