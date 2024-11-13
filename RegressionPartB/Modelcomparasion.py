import numpy as np
from sklearn import model_selection
from ucimlrepo import fetch_ucirepo
import pandas as pd
import inspect
import torch
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

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
y = stats.zscore(y)

k1 = k2 = 10
hiddenUnits = 1
n_replicates = 1  # number of networks trained in each k-fold
max_iter = 10000

lambdas = np.logspace(-3, 0, 10)

CV1 = model_selection.KFold(k1, shuffle=True)
for i, (D_par_i, D_test_i) in enumerate(CV1.split(X, y)):
    ANNerrors=[]
    reglinerrors=[]
    baseerrors=[]
    bestNN = 0
    bestLambda = 0
    CV2 = model_selection.KFold(k2, shuffle=True)
    for j, (D_train_i, D_val_i) in enumerate(CV2.split(X[D_par_i], y[D_par_i])):
        
        ANNinnermodel = lambda: torch.nn.Sequential(
            torch.nn.Linear(M, j+hiddenUnits),  # M features to n_hidden_units
            torch.nn.Tanh(),  # 1st transfer function,
            torch.nn.Linear(j+hiddenUnits, 1),  # n_hidden_units to 1 output neuron
            # no final tranfer function, i.e. "linear output"
        )
        loss_fn = torch.nn.MSELoss()

        AnnX_train = torch.Tensor(X[D_train_i, :])
        ANNy_train = torch.Tensor(y[D_train_i])
        ANNX_val = torch.Tensor(X[D_val_i, :])
        ANNy_val = torch.Tensor(y[D_val_i])

        innernet, final_loss, learning_curve = train_neural_net(
        ANNinnermodel,
        loss_fn,
        X=AnnX_train,
        y=ANNy_train,
        n_replicates=n_replicates,
        max_iter=max_iter,
        )
        # Determine estimated class labels for test set
        ANNy_val_est = innernet(ANNX_val)
        
        # Determine ANNerrors and ANNerrors
        se = (ANNy_val_est.float() - ANNy_val.float()) ** 2  # squared error
        ANNmse = (sum(se).type(torch.float) / len(ANNy_val)).data.numpy()  # mean
        ANNerrors.append(ANNmse)
        if (len(ANNerrors)>1 and ANNerrors[j]<ANNerrors[bestNN]): bestNN = j

        linX_train = (X[D_train_i, :])
        liny_train = (y[D_train_i])
        linX_val = (X[D_val_i, :])
        liny_val = (y[D_val_i])

        ridge = Ridge(alpha=lambdas[j], fit_intercept=False)
        ridge.fit(linX_train, liny_train)
        
        linmse = mean_squared_error(liny_val, ridge.predict(linX_val))  # mean
        
        reglinerrors.append(linmse)
        if (reglinerrors[j]<reglinerrors[bestLambda]): bestLambda = j

    ANNX_par = torch.Tensor(X[D_par_i, :])
    ANNy_par = torch.Tensor(y[D_par_i])
    ANNX_test = torch.Tensor(X[D_test_i, :])
    ANNy_test = torch.Tensor(y[D_test_i])
    
    ANNoutermodel = lambda: torch.nn.Sequential(
        torch.nn.Linear(M, bestNN+hiddenUnits),  # M features to n_hidden_units
        torch.nn.Tanh(),  # 1st transfer function,
        torch.nn.Linear(bestNN+hiddenUnits, 1),  # n_hidden_units to 1 output neuron
        # no final tranfer function, i.e. "linear output"
    )
    loss_fn = torch.nn.MSELoss()

    outernet, final_loss, learning_curve = train_neural_net(
    ANNoutermodel,
    loss_fn,
    X=ANNX_par,
    y=ANNy_par,
    n_replicates=n_replicates,
    max_iter=max_iter,
    )
    # Determine estimated class labels for test set
    ANNy_test_est = outernet(ANNX_test)
    se = (ANNy_test_est.float() - ANNy_test.float()) ** 2  # squared error
    ANNmse = (sum(se).type(torch.float) / len(ANNy_test)).data.numpy()  # mean

    print("ANN:\nOuter fold:",i,"\tHidden units",bestNN+hiddenUnits,"\tEtest_{0}".format(i),ANNmse)

    linX_par = (X[D_par_i, :])
    liny_par = (y[D_par_i])
    linX_test = (X[D_test_i, :])
    liny_test = (y[D_test_i])
    
    ridge = Ridge(alpha=lambdas[bestLambda], fit_intercept=False)
    ridge.fit(linX_par, liny_par)
    
    linmse = mean_squared_error(liny_test, ridge.predict(linX_test))  # mean

    print("Linear Reg\nOuter fold:",i,"\tLambda",lambdas[bestLambda],"\tEtest_{0}".format(i),linmse)

    baseline = np.repeat(liny_par.mean(), len(liny_test))
        
    mse = mean_squared_error(liny_test, baseline)

    print("Baseline\nOuter fold:",i,"\tEtest_{0}".format(i),mse)
        