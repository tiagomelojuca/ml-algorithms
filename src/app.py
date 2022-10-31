import math

import matplotlib.pyplot as plt
import numpy as np

import mlfa
import test_cases

def g(i):
    return mlfa._tanh(i)

def g_(i):
    return mlfa._ddxtanh_vec(i)

def forward(LAYERS_W, LAYERS_I, LAYERS_Y, x):
    j = 0
    for j in range(len(LAYERS_W)):
        if j == 0:
            LAYERS_I[j] = LAYERS_W[j] @ x
        else:
            ybias = mlfa._addLineOfValueTo(-1, LAYERS_Y[j - 1])
            LAYERS_I[j] = LAYERS_W[j] @ ybias
        LAYERS_Y[j] = g(LAYERS_I[j])
        j += 1
    return LAYERS_W, LAYERS_I, LAYERS_Y

def backward(LAYERS_W, LAYERS_I, LAYERS_Y, LAYERS_D, LR, x, d):
    j = len(LAYERS_W) - 1
    while j >= 0:
        if j + 1 == len(LAYERS_W):
            LAYERS_D[j] = mlfa._hadamard_product(g_(LAYERS_I[j]), (d - LAYERS_Y[j]))
            ybias = mlfa._addLineOfValueTo(-1, LAYERS_Y[j - 1])
            LAYERS_W[j] = LAYERS_W[j] + LR * mlfa._outer_product(LAYERS_D[j], ybias)
        elif j == 0:
            _W = LAYERS_W[j + 1]
            _W_without_bias = _W[:, 1:]
            Wb = _W_without_bias.T
            LAYERS_D[j] = mlfa._hadamard_product(g_(LAYERS_I[j]), (Wb @ LAYERS_D[j + 1]))
            LAYERS_W[j] = LAYERS_W[j] + LR * mlfa._outer_product(LAYERS_D[j], x)
        else:
            _W = LAYERS_W[j + 1]
            _W_without_bias = _W[:, 1:]
            Wb = _W_without_bias.T
            LAYERS_D[j] = mlfa._hadamard_product(g_(LAYERS_I[j]), (Wb @ LAYERS_D[j + 1]))
            ybias = mlfa._addLineOfValueTo(-1, LAYERS_Y[j - 1])
            LAYERS_W[j] = LAYERS_W[j] + LR * mlfa._outer_product(LAYERS_D[j], ybias)
        j -= 1

    return LAYERS_W, LAYERS_I, LAYERS_Y, LAYERS_D

def EQM(LAYERS_W, LAYERS_I, LAYERS_Y, X, Y):
    numberOfSamples = X.shape[1]

    eqm = 0
    for i in range(numberOfSamples):
        x = X[:, i]
        x.shape = (x.shape[0], 1)
        _, LAYERS_I, LAYERS_Y = forward(LAYERS_W, LAYERS_I, LAYERS_Y, x)
        d = Y[:, i]
        d.shape = (d.shape[0], 1)
        eqi = 0
        neurons_output_layer = LAYERS_I[-1].shape[0]
        _Y = LAYERS_Y[-1]
        j = 0
        for j in range(neurons_output_layer):
            err = d[j, 0] - _Y[j, 0]
            eqi += err * err
            j += 1
        eqm += eqi
    eqm /= 2 * numberOfSamples

    return LAYERS_W, LAYERS_I, LAYERS_Y, eqm

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

L = 3
INPUT_LAYER_SIGNALS = X.shape[1]
NEURONS_PER_LAYER   = [2, 4, 3, 5]
NUM_LAYERS = len(NEURONS_PER_LAYER)
LR = 0.001
ME = 1000
PREC = 0.1

LAYERS_W = []
LAYERS_I = []
LAYERS_Y = []
LAYERS_D = []

for i in range(NUM_LAYERS):
    neurons_current_layer = NEURONS_PER_LAYER[i]

    num_inputs_without_bias = INPUT_LAYER_SIGNALS if i == 0 else NEURONS_PER_LAYER[i - 1]
    num_inputs = num_inputs_without_bias + 1
    LAYERS_W.append(np.random.random_sample((neurons_current_layer, num_inputs)) - 0.5)

    LAYERS_I.append(np.empty((neurons_current_layer, 1)))
    LAYERS_Y.append(np.empty((neurons_current_layer, 1)))
    LAYERS_D.append(np.empty((neurons_current_layer, 1)))

# =======================================================================================

Xtrain = mlfa._addLineOfValueTo(-1, Xtrain)

eqm = 1
t = 0
while eqm > PREC and t < ME:
    numSamples = Xtrain.shape[1]
    for i in range(numSamples):
        x = Xtrain[:, i]
        x.shape = (x.shape[0], 1)
        _, LAYERS_I, LAYERS_Y = forward(LAYERS_W, LAYERS_I, LAYERS_Y, x)
        d = Ytrain[:, i]
        d.shape = (d.shape[0], 1)
        LAYERS_W, LAYERS_I, LAYERS_Y, LAYERS_D = backward(LAYERS_W, LAYERS_I, LAYERS_Y, LAYERS_D, LR, x, d)
    _, LAYERS_I, LAYERS_Y, eqm = EQM(LAYERS_W, LAYERS_I, LAYERS_Y, Xtrain, Ytrain)
    t += 1

# =======================================================================================

Ypred = np.empty((0, 5))
for i in range(Xtest.shape[0]):
    x = Xtest[i, :]
    x.shape = (1, Xtest.shape[1])
    x = mlfa._addLineOfValueTo(-1, x.T)
    _, _, _Y = forward(LAYERS_W, LAYERS_I, LAYERS_Y, x)
    _Ypred = _Y[-1].T
    Ypred = np.concatenate((Ypred, _Ypred), axis=0)
print("Accuracy : " + str(mlfa._calcAccuracy(Ytest, Ypred)))

# =======================================================================================

print("Goodbye, World!")
