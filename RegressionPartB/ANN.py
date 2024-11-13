import numpy as np
from sklearn import model_selection
from ucimlrepo import fetch_ucirepo
import pandas as pd
import inspect
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

# Normalize data
X = stats.zscore(X)

k1 = k2 = 10
hiddenUnits = 20
n_replicates = 1  # number of networks trained in each k-fold
max_iter = 10000

CV1 = model_selection.KFold(k1, shuffle=True)
for i, (D_par_i, D_test_i) in enumerate(CV1.split(X, y)):
    errors=[]
    bestNN = 0
    CV2 = model_selection.KFold(k2, shuffle=True)
    for j, (D_train_i, D_val_i) in enumerate(CV2.split(X[D_par_i], y[D_par_i])):
        
        model = lambda: torch.nn.Sequential(
            torch.nn.Linear(M, j+hiddenUnits),  # M features to n_hidden_units
            torch.nn.Tanh(),  # 1st transfer function,
            torch.nn.Linear(j+hiddenUnits, 1),  # n_hidden_units to 1 output neuron
            # no final tranfer function, i.e. "linear output"
        )
        loss_fn = torch.nn.MSELoss()

        X_train = torch.Tensor(X[D_train_i, :])
        y_train = torch.Tensor(y[D_train_i])
        X_val = torch.Tensor(X[D_val_i, :])
        y_val = torch.Tensor(y[D_val_i])

        net, final_loss, learning_curve = train_neural_net(
        model,
        loss_fn,
        X=X_train,
        y=y_train,
        n_replicates=n_replicates,
        max_iter=max_iter,
        )
        # Determine estimated class labels for test set
        y_val_est = net(X_val)
        
        # Determine errors and errors
        se = (y_val_est.float() - y_val.float()) ** 2  # squared error
        mse = (sum(se).type(torch.float) / len(y_val)).data.numpy()  # mean
        errors.append(mse)
        if (len(errors)>1 and errors[j]<errors[bestNN]): bestNN = j

    X_par = torch.Tensor(X[D_par_i, :])
    y_par = torch.Tensor(y[D_par_i])
    X_test = torch.Tensor(X[D_test_i, :])
    y_test = torch.Tensor(y[D_test_i])
    
    model = lambda: torch.nn.Sequential(
        torch.nn.Linear(M, bestNN+hiddenUnits),  # M features to n_hidden_units
        torch.nn.Tanh(),  # 1st transfer function,
        torch.nn.Linear(bestNN+hiddenUnits, 1),  # n_hidden_units to 1 output neuron
        # no final tranfer function, i.e. "linear output"
    )
    loss_fn = torch.nn.MSELoss()

    net, final_loss, learning_curve = train_neural_net(
    model,
    loss_fn,
    X=X_train,
    y=y_train,
    n_replicates=n_replicates,
    max_iter=max_iter,
    )
    # Determine estimated class labels for test set
    y_test_est = net(X_test)
    se = (y_test_est.float() - y_test.float()) ** 2  # squared error
    mse = (sum(se).type(torch.float) / len(y_test)).data.numpy()  # mean

    print("\nOuter fold:",i,"\tHidden units",bestNN+hiddenUnits,"\tEtest_{0}".format(i),mse)
        