import math

import matplotlib.pyplot as plt
import numpy as np

import mlfa
import test_cases

# =======================================================================================

X, Y = test_cases.DUMMYMULTICLS_getData()
Xtrain, Ytrain, Xtest, Ytest = mlfa._prepareRound(X, Y)
Wlist = mlfa.W_MLP(Xtrain, Ytrain, [2, 4, 3], 5, 0.001, 1000, 0.1)
Ypred = mlfa.Y_MLP(Wlist, Xtest)
print("Accuracy : " + str(mlfa._calcAccuracy(Ytest, Ypred)))

# =======================================================================================

print("Goodbye, World!")
