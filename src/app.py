import time

import numpy as np

import mlfa
import test_cases

# =======================================================================================

def _computeStatistics(results):
    results_times = results[:, 0]
    results_times.shape = (results_times.shape[0], 1)

    results_accs  = results[:, 1]
    results_accs.shape  = (results_accs.shape[0],  1)

    times_min  = results_times.min()
    times_max  = results_times.max()
    times_mean = np.mean(results_times, axis = 0)

    accs_min   = results_accs.min()
    accs_max   = results_accs.max()
    accs_mean  = np.mean(results_accs, axis = 0)

    computedStatistics = np.empty((2, 3))
    computedStatistics[0, 0] = times_min
    computedStatistics[0, 1] = times_mean
    computedStatistics[0, 2] = times_max
    computedStatistics[1, 0] = accs_min
    computedStatistics[1, 1] = accs_mean
    computedStatistics[1, 2] = accs_max

    return computedStatistics

# =======================================================================================

def _printStatistics(header, results):
    computedStatistics = _computeStatistics(results)
    print("# ==================================================")
    print("# " + header)
    print("# ==================================================")
    print("Time [MIN]      : %.2f s"  % computedStatistics[0, 0])
    print("Time [MEAN]     : %.2f s"  % computedStatistics[0, 1])
    print("Time [MAX]      : %.2f s"  % computedStatistics[0, 2])
    print("Accuracy [MIN]  : %.2f %%" % computedStatistics[1, 0])
    print("Accuracy [MEAN] : %.2f %%" % computedStatistics[1, 1])
    print("Accuracy [MAX]  : %.2f %%" % computedStatistics[1, 2])
    print("# ==================================================")

# =======================================================================================
# DATASET
# =======================================================================================

NUMBER_OF_ROUNDS = 100 # results = np.empty((NoF, 2)), where [i, 0] == time and [i, 1] == acc
X, Y = test_cases.CMUFACEIMAGES_getData(30) # 30x30 = 900 features
X = X.T
Y = Y.T

# =======================================================================================
# OLS
# =======================================================================================

resultsOLS = np.empty((NUMBER_OF_ROUNDS, 2))
for i in range(NUMBER_OF_ROUNDS):
    Xtrain, Ytrain, Xtest, Ytest = mlfa._prepareRound(X, Y)
    start = time.time()
    W = mlfa.W_OLS(Xtrain, Ytrain)
    end   = time.time()
    Ypred = mlfa.Y_OLS(W, Xtest)
    timeElapsed = end - start # TE
    accuracy = mlfa._calcAccuracy(Ytest, Ypred)
    resultsOLS[i, 0] = timeElapsed
    resultsOLS[i, 1] = accuracy

# =======================================================================================
# OLS_reg
# =======================================================================================

resultsOLS_reg = np.empty((NUMBER_OF_ROUNDS, 2))
for i in range(NUMBER_OF_ROUNDS):
    Xtrain, Ytrain, Xtest, Ytest = mlfa._prepareRound(X, Y)
    start = time.time()
    W = mlfa.W_OLS(Xtrain, Ytrain, 2)
    end   = time.time()
    Ypred = mlfa.Y_OLS(W, Xtest)
    timeElapsed = end - start # TE
    accuracy = mlfa._calcAccuracy(Ytest, Ypred)
    resultsOLS_reg[i, 0] = timeElapsed
    resultsOLS_reg[i, 1] = accuracy

# =======================================================================================
# SLP
# =======================================================================================

resultsSLP = np.empty((NUMBER_OF_ROUNDS, 2))
for i in range(NUMBER_OF_ROUNDS):
    Xtrain, Ytrain, Xtest, Ytest = mlfa._prepareRound(X, Y)
    start = time.time()
    W = mlfa.W_SLP(Xtrain, Ytrain, 0.01, 1000)
    end   = time.time()
    Ypred = mlfa.Y_OLS(W, Xtest)
    timeElapsed = end - start # TE
    accuracy = mlfa._calcAccuracy(Ytest, Ypred)
    resultsSLP[i, 0] = timeElapsed
    resultsSLP[i, 1] = accuracy

# =======================================================================================
# MLP
# =======================================================================================

X = mlfa._normalize_samples_minmax(X)

resultsMLP = np.empty((NUMBER_OF_ROUNDS, 2))
for i in range(NUMBER_OF_ROUNDS):
    Xtrain, Ytrain, Xtest, Ytest = mlfa._prepareRound(X, Y)
    start = time.time()
    Wlist = mlfa.W_MLP(Xtrain, Ytrain, [100], 20, 0.01, 1000, 0.6)
    end   = time.time()
    Ypred = mlfa.Y_MLP(Wlist, Xtest)
    timeElapsed = end - start # TE
    accuracy = mlfa._calcAccuracy(Ytest, Ypred)
    resultsMLP[i, 0] = timeElapsed
    resultsMLP[i, 1] = accuracy

# =======================================================================================

_printStatistics("OLS", resultsOLS)
_printStatistics("OLS_reg", resultsOLS_reg)
_printStatistics("SLP", resultsSLP)
_printStatistics("MLP", resultsMLP)

# =======================================================================================
