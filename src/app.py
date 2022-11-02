import math

import matplotlib.pyplot as plt
import numpy as np

import mlfa
import test_cases

def normalize(X):
    for i in range(X.shape[0]):
        x = X[i, :]
        x.shape = (1, X.shape[1])
        norma = math.sqrt(x[0, 0] * x[0, 0] + x[0, 1] * x[0, 1])
        x[0, 0] /= norma
        x[0, 1] /= norma
    return X

# =======================================================================================

X, Y = test_cases.DUMMYMULTICLS_getData()
# test_cases.DUMMYMULTICLS_showData()
# X = normalize(X)
Xtrain, Ytrain, Xtest, Ytest = mlfa._prepareRound(X, Y)
Xtrain = Xtrain.T
Ytrain = Ytrain.T

# =======================================================================================

Xtrain = mlfa._addLineOfValueTo(-1, Xtrain)
Wlist = mlfa._W_MLP(Xtrain, Ytrain, [2, 4, 3], 5, 0.001, 1000, 0.1)
Ypred = mlfa._Y_MLP(Xtest, Wlist)
print("Accuracy : " + str(mlfa._calcAccuracy(Ytest, Ypred)))

# =======================================================================================

print("Goodbye, World!")
