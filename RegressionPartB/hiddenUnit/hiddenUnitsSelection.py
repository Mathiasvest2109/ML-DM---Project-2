import importlib_resources
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.io import loadmat
from sklearn import model_selection
from ucimlrepo import fetch_ucirepo


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

genError=[] #make list for generalization error for each value of h

hiddenlayerrange=31

for i in range(1,hiddenlayerrange):
    print("\nHIDDEN UNITS{0}\n".format(i))
    # Parameters for neural network classifier
    n_hidden_units = i  # number of hidden units
    n_replicates = 1  # number of networks trained in each k-fold
    max_iter = 10000

    # K-fold crossvalidation
    K = 3  # only three folds to speed up this example
    CV = model_selection.KFold(K, shuffle=True)

    # Setup figure for display of learning curves and error rates in fold
    #summaries, summaries_axes = plt.subplots(1, 2, figsize=(10, 5))
    # Make a list for storing assigned color of learning curve for up to K=10
    color_list = [
        "tab:orange",
        "tab:green",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
        "tab:red",
        "tab:blue",
    ]
    # Define the model
    model = lambda: torch.nn.Sequential(
        torch.nn.Linear(M, n_hidden_units),  # M features to n_hidden_units
        torch.nn.Tanh(),  # 1st transfer function,
        torch.nn.Linear(n_hidden_units, 1),  # n_hidden_units to 1 output neuron
        # no final tranfer function, i.e. "linear output"
    )
    loss_fn = torch.nn.MSELoss()  # notice how this is now a mean-squared-error loss

    print("Training model of type:\n\n{}\n".format(str(model())))
    errors = []  # make a list for storing generalizaition error in each loop
    for k, (train_index, test_index) in enumerate(CV.split(X, y)):
        print("\nCrossvalidation fold: {0}/{1}".format(k + 1, K))

        # Extract training and test set for current CV fold, convert to tensors
        X_train = torch.Tensor(X[train_index, :])
        y_train = torch.Tensor(y[train_index])
        X_test = torch.Tensor(X[test_index, :])
        y_test = torch.Tensor(y[test_index])

        # Train the net on training data
        net, final_loss, learning_curve = train_neural_net(
            model,
            loss_fn,
            X=X_train,
            y=y_train,
            n_replicates=n_replicates,
            max_iter=max_iter,
        )

        print("\n\tBest loss: {}\n".format(final_loss))

        # Determine estimated class labels for test set
        y_test_est = net(X_test)

        # Determine errors and errors
        se = (y_test_est.float() - y_test.float()) ** 2  # squared error
        mse = (sum(se).type(torch.float) / len(y_test)).data.numpy()  # mean
        errors.append(mse)  # store error rate for current CV fold

        # Display the learning curve for the best net in the current fold
        # (h,) = summaries_axes[0].plot(learning_curve, color=color_list[k])
        # h.set_label("CV fold {0}".format(k + 1))
        # summaries_axes[0].set_xlabel("Iterations")
        # summaries_axes[0].set_xlim((0, max_iter))
        # summaries_axes[0].set_ylabel("Loss")
        # summaries_axes[0].set_title("Learning curves")

    # Display the MSE across folds
    # summaries_axes[1].bar(
    #     np.arange(1, K + 1), np.squeeze(np.asarray(errors)), color=color_list
    # )
    # summaries_axes[1].set_xlabel("Fold")
    # summaries_axes[1].set_xticks(np.arange(1, K + 1))
    # summaries_axes[1].set_ylabel("MSE")
    # summaries_axes[1].set_title("Test mean-squared-error")
    print("Diagram of best neural net in last fold:")
    weights = [net[i].weight.data.numpy().T for i in [0, 2]]
    biases = [net[i].bias.data.numpy() for i in [0, 2]]
    tf = [str(net[i]) for i in [1, 2]]
    #draw_neural_net(weights, biases, tf, attribute_names=attributeNames)

    # Print the average classification error rate
    print(
        "\nEstimated generalization error, RMSE: {0}".format(
            round(np.sqrt(np.mean(errors)), 4)
        )
    )

    # Print the average classification error rate
    genError.append(round(np.sqrt(np.mean(errors)), 4))
    print(
        "\nGeneralization error/average error rate: {0}%".format(
            genError[i-1]
        )
    )

for i in range(1,hiddenlayerrange):
    print(
        "\nAmount of hidden units: {0}\nGeneralization error: {1}".format(
            i,genError[i-1]
        )
    )


plt.plot(range(1,hiddenlayerrange),genError, 'o')
plt.xlabel("number of hidden units")
plt.ylabel("Generalization error")
plt.show()