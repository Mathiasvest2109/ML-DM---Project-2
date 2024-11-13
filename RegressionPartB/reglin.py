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

k1 = k2 = 10
lambdas = np.logspace(-3, 0, 10)

CV1 = model_selection.KFold(k1, shuffle=True)
for i, (D_par_i, D_test_i) in enumerate(CV1.split(X, y)):
    reglinerrors=[]
    bestLambda = 0
    CV2 = model_selection.KFold(k2, shuffle=True)
    for j, (D_train_i, D_val_i) in enumerate(CV2.split(X[D_par_i], y[D_par_i])):

        linX_train = (X[D_train_i, :])
        liny_train = (y[D_train_i])
        linX_val = (X[D_val_i, :])
        liny_val = (y[D_val_i])

        ridge = Ridge(alpha=lambdas[j], fit_intercept=False)
        ridge.fit(linX_train, liny_train)
        
        linmse = mean_squared_error(liny_val, ridge.predict(linX_val))  # mean
        
        reglinerrors.append(linmse)
        if (reglinerrors[j]<reglinerrors[bestLambda]): bestLambda = j
    linX_par = (X[D_par_i, :])
    liny_par = (y[D_par_i])
    linX_test = (X[D_test_i, :])
    liny_test = (y[D_test_i])
    
    ridge = Ridge(alpha=lambdas[bestLambda], fit_intercept=False)
    ridge.fit(linX_par, liny_par)
    
    linmse = mean_squared_error(liny_test, ridge.predict(linX_test))  # mean

    print("\nOuter fold:",i,"\tLambda",lambdas[bestLambda],"\tEtest_{0}".format(i),linmse)
        