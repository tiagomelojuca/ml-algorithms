###############################################################################
# MLFA (Machine Learning Fundamental Algorithms
#
# Please note that this code looks like a piece of s***. I have never used
# Python before, and it's just a exercise. I'm putting no effort at all in
# the code, and have no interest about how a Python code should like, or
# "how pythonic" it is. The goal here is learn the ML concept.
# So, it's just for studying purposes. But feel free to study with me :)
#
# You'll see a lot of comments here. I usually use them as remainder for
# myself to search about the subject later. Also, you'll note some "?"
# when I'm not sure if it fits a particular concept
###############################################################################
import sys
import math
import numpy as np
###############################################################################
# PUBLIC ALGORITHMS FOR TRAINING THE MODEL
#
# X  : Inputs Matrix [N x P]
#      N stands for Number of Samples
#      P stands for Number of Predictors
#      each row is an input vector (of a sample), for which
#      we know the expected/desired outcome
# Y  : Labels Matrix [N x C]
#      N stands for Number of Samples
#      C stands for Number of Classes
#      each row is a label (of a sample), that must be a vector in which
#      all elements are -1, except by the index of its respective class
# LR : Learning Rate (0 < LR <= 1)
#      Hyperparameter for adjust the learning rule
# ME : Max Epochs (ME > 0)
#      Hyperparameter for force the end of training
# W  : Weights Matrix [(P + 1) x C]
#      P stands for Number of Predictors
#      C stands for Number of Classes
#      the estimated matrix of weights for the model,
#      which will be used for performing predictions by the model
###############################################################################

# Estimate W by Ordinary Least Squares
def W_OLS(X, Y, RP = -1):
    _X = _addColumnOfValueTo(1, X)
    if RP < 0:
        return _W_OLS_Calculate(_X, Y)
    else:
        return _W_OLS_Tikhonov_Regularization(_X, Y, RP)

###############################################################################

# Estimate W by Single-Layer Perceptron
def W_SLP(X, Y, LR, ME):
    return _W_SLP_Multiclass(X, Y, LR, ME)

###############################################################################

# Estimate W by Multi-Layer Perceptron
def W_MLP(X, Y, NHL, NOL, LR, ME, TOL):
    X = _normalize_samples_minmax(X)
    X = X.T
    Y = Y.T
    X = _addLineOfValueTo(-1, X)
    weightsLayers = _W_MLP(X, Y, NHL, NOL, LR, ME, TOL)

    return weightsLayers

###############################################################################
# PUBLIC ALGORITHMS FOR PERFORMING PREDICTIONS
#
# W : Weights Matrix [(P + 1) x C]
#     P stands for Number of Predictors
#     C stands for Number of Classes
# X : Inputs Matrix [N x P]
#     N stands for Number of Inputs for performing predictions
#     P stands for Number of Predictors
#     each row is an input vector, however now it's not known sample anymore,
#     but instead an input we would like to predict the outcome
# Y : Outputs Matrix [N x C]
#     N stands for Number of Predicted Samples
#     C stands for Number of Classes
#     each row is an output vector, a prediction performed by the model
#     for its respective input vector (so that we have a predicted sample)
###############################################################################

# Estimate Y by Ordinary Least Squares
def Y_OLS(W, X):
    _X = _addColumnOfValueTo(1, X)
    return _X @ W # Ypred

###############################################################################

# Estimate Y by Single-Layer Perceptron
def Y_SLP(W, X):
    _X = _addColumnOfValueTo(-1, X)
    return _X @ W # Ypred

###############################################################################

# Estimate Y by Multi-Layer Perceptron
def Y_MLP(W, X):
    return _Y_MLP(W, X)

###############################################################################
# PRIVATE ALGORITHMS FOR THE ACTUAL BUSINESS LOGIC
#
# The public interface above are just some wrappers for standardize the API
# Below is the core logic, where the hard work is actually done (heavy work?)
###############################################################################

# Regression applying the formula of Ordinary Least Squares (OLS)
# is is the same thing as Least Mean Square (LMS)? Seens so
# X : Input Matrix   [N x (P + 1)], already with column of ones
# Y : Label Matrix   [N x C]
# W : Weights Matrix [(P + 1) x C]
def _W_OLS_Calculate(X, Y):
    return np.linalg.pinv(X.T @ X) @ X.T @ Y

# HP : hyperparam (ridge parameter (RP) >= 0)
def _W_OLS_Tikhonov_Regularization(X, Y, HP):
    XtX = X.T @ X
    I = np.identity(XtX.shape[0])
    return np.linalg.pinv(XtX + HP * I) @ X.T @ Y

###############################################################################

# Binary classification with Perceptron (Neuron MP vs Rosenblatt? Only one neuron?)
# (aka PS, Perceptron Simples / [Simple Perceptron?])
# X        : Input Matrix   [N x P]
# Y        : Label Matrix   [N x C]
# LR       : 0 < LR <= 1, stands for the Learning Rate, hyperparameter (aka n? aka r?)
# maxEpoch : maxEpoch > 0, give up if reach the max number of epochs, hyperparameter
# W        : Weights Matrix [(P + 1) x C]
# t        : epochs at end of training
def _W_SLP_Binary(X, Y, LR, maxEpoch):
    X = X.T
    X = _addLineOfValueTo(-1, X)
    numSamples = X.shape[1]

    W = np.zeros((X.shape[0], 1))
    hasError = True # ofc it does, our W is still just zeros
    
    t = 0 # EPOCH
    while hasError == True and t <= maxEpoch:
        hasError = False

        for i in range(numSamples):
            x_t = X[:, i]
            x_t.shape = (x_t.shape[0], 1)

            u_t = W.T @ x_t
            y_t = _shl(u_t)
            
            d_t = int(Y[i, 0])

            W = W + LR * (d_t - y_t) * x_t / 2

            if d_t != y_t:
                hasError = True

        t += 1

    return (W, t)

###############################################################################

# Multiclass classification with OvR plus the binary classification perceptron
# (OVR stands for One vs Rest) (aka OvA, One vs All? same thing?)
# single layer perceptron can have more than one neuron (at same layer)
# not sure what it means, maybe it is this case? if so, is this still a
# Simple Perceptron (SP)? actually, even if not being the case, is it still a SP?
def _W_SLP_Multiclass(X, Y, LR, maxEpoch):
    p = X.shape[1] # number of (predictors?|features?)
    W = np.empty((p + 1, 0))

    numberOfClasses = Y.shape[1]
    for i in range(numberOfClasses):
        _W, t = _W_SLP_Binary(X, _y(Y, i), LR, maxEpoch)
        W = np.concatenate((W, _W), axis = 1)
    
    return W

###############################################################################

# X : (P + 1) x N
# Y : C x N
# NHL : neuronsHiddenLayers
# NOL : neuronsOutputLayers
# TOL : tolerance / prec(ision)
def _W_MLP(X, Y, NHL, NOL, LR, ME, TOL):
    numEntriesIL, numLayers, neuronsLayers = _W_MLP_parseInfo(X, NHL, NOL)
    lW, lI, lY, lD = _W_MLP_initLayers(numLayers, neuronsLayers, numEntriesIL)
    lW, lI, lY, lD = _W_MLP_train(X, Y, LR, ME, TOL, lW, lI, lY, lD)

    return lW

def _W_MLP_parseInfo(X, NHL, NOL):
    numberOfSignalsInputLayer = X.shape[0] - 1 # minus 1 due to bias
    neuronsPerLayer = NHL.copy()
    neuronsPerLayer.append(NOL)
    numberOfLayers = len(neuronsPerLayer)

    return numberOfSignalsInputLayer, numberOfLayers, neuronsPerLayer

def _W_MLP_initLayers(numberOfLayers, neuronsPerLayer, numberOfSignalsInputLayer):
    LAYERS_W = []
    LAYERS_I = []
    LAYERS_Y = []
    LAYERS_D = []

    for i in range(numberOfLayers):
        neuronsCurrentLayer  = neuronsPerLayer[i]
        numInputsWithoutBias = numberOfSignalsInputLayer if i == 0 else neuronsPerLayer[i - 1]
        numInputs            = numInputsWithoutBias + 1

        LAYERS_W.append(np.random.random_sample((neuronsCurrentLayer, numInputs)) - 0.5)
        LAYERS_I.append(np.empty((neuronsCurrentLayer, 1)))
        LAYERS_Y.append(np.empty((neuronsCurrentLayer, 1)))
        LAYERS_D.append(np.empty((neuronsCurrentLayer, 1)))

    return LAYERS_W, LAYERS_I, LAYERS_Y, LAYERS_D

def _W_MLP_train(X, Y, LR, ME, TOL, layersW, layersI, layersY, layersD):
    eqm = 1
    t = 0
    while eqm > TOL and t < ME:
        numSamples = X.shape[1]
        for i in range(numSamples):
            x = X[:, i]
            x.shape = (x.shape[0], 1)
            _, layersI, layersY = _W_MLP_forward(layersW, layersI, layersY, x)
            d = Y[:, i]
            d.shape = (d.shape[0], 1)
            layersW, layersI, layersY, layersD = _W_MLP_backward(layersW, layersI, layersY, layersD, LR, x, d)
        _, layersI, layersY, eqm = _W_MLP_EQM(layersW, layersI, layersY, X, Y)
        t += 1
    
    return layersW, layersI, layersY, layersD

def _W_MLP_forward(layersW, layersI, layersY, x):
    j = 0
    for j in range(len(layersW)):
        if j == 0:
            layersI[j] = layersW[j] @ x
        else:
            ybias = _addLineOfValueTo(-1, layersY[j - 1])
            layersI[j] = layersW[j] @ ybias
        layersY[j] = _W_MLP_g(layersI[j])
        j += 1
    return layersW, layersI, layersY

def _W_MLP_backward(layersW, layersI, layersY, layersD, LR, x, d):
    j = len(layersW) - 1
    while j >= 0:
        if j + 1 == len(layersW):
            layersD[j] = _hadamard_product(_W_MLP_ddxg(layersI[j]), (d - layersY[j]))
            ybias = _addLineOfValueTo(-1, layersY[j - 1])
            layersW[j] = layersW[j] + LR * _outer_product(layersD[j], ybias)
        elif j == 0:
            _W = layersW[j + 1]
            _W_without_bias = _W[:, 1:]
            Wb = _W_without_bias.T
            layersD[j] = _hadamard_product(_W_MLP_ddxg(layersI[j]), (Wb @ layersD[j + 1]))
            layersW[j] = layersW[j] + LR * _outer_product(layersD[j], x)
        else:
            _W = layersW[j + 1]
            _W_without_bias = _W[:, 1:]
            Wb = _W_without_bias.T
            layersD[j] = _hadamard_product(_W_MLP_ddxg(layersI[j]), (Wb @ layersD[j + 1]))
            ybias = _addLineOfValueTo(-1, layersY[j - 1])
            layersW[j] = layersW[j] + LR * _outer_product(layersD[j], ybias)
        j -= 1

    return layersW, layersI, layersY, layersD

# mean squared error (MSE)
def _W_MLP_EQM(layersW, layersI, layersY, X, Y):
    numberOfSamples = X.shape[1]

    eqm = 0
    for i in range(numberOfSamples):
        x = X[:, i]
        x.shape = (x.shape[0], 1)
        _, layersI, layersY = _W_MLP_forward(layersW, layersI, layersY, x)
        d = Y[:, i]
        d.shape = (d.shape[0], 1)
        eqi = 0
        neuronsOutputLayer = layersI[-1].shape[0]
        _y = layersY[-1]
        j = 0
        for j in range(neuronsOutputLayer):
            err = d[j, 0] - _y[j, 0]
            eqi += err * err
            j += 1
        eqm += eqi
    eqm /= 2 * numberOfSamples

    return layersW, layersI, layersY, eqm

def _W_MLP_g(i):
    return _tanh_vec(i)

def _W_MLP_ddxg(i):
    return _ddxtanh_vec(i)

# X : N x P
def _Y_MLP(layersW, X):
    layersI, layersY = _Y_MLP_initLayers(layersW)

    Ypred = np.empty((0, 5))
    for i in range(X.shape[0]):
        x = X[i, :]
        x.shape = (1, X.shape[1])
        x = _addLineOfValueTo(-1, x.T)
        _, _, _Y = _W_MLP_forward(layersW, layersI, layersY, x)
        _Ypred = _Y[-1].T
        Ypred = np.concatenate((Ypred, _Ypred), axis=0)

    return Ypred

def _Y_MLP_initLayers(layersW):
    layersI = []
    layersY = []
    
    for i in range(len(layersW)):
        neuronsCurrentLayer = layersW[i].shape[0]
        layersI.append(np.empty((neuronsCurrentLayer, 1)))
        layersY.append(np.empty((neuronsCurrentLayer, 1)))

    return layersI, layersY

###############################################################################
# HELPERS SHOULD BE PRIVATE...
###############################################################################

def _addLineOfValueTo(value, X):
    ones = np.ones((1, X.shape[1]))
    line = value * ones
    return np.concatenate((line, X), axis = 0)

###############################################################################

def _addColumnOfValueTo(value, X):
    ones = np.ones((X.shape[0], 1))
    column = value * ones
    return np.concatenate((column, X), axis = 1)

###############################################################################

def _prepareRound(X, Y, percentTrain = 0.8):
    if percentTrain <= 0.0 or percentTrain >= 1.0:
        raise Exception("Percent of training samples must be within range (0.0, 1.0)")
    
    seed            = np.random.permutation(X.shape[0])
    Xscrambled      = X[seed, :]
    Yscrambled      = Y[seed, :]
    samplesTraining = int(Xscrambled.shape[0] * percentTrain)

    Xtrain = Xscrambled[0 : samplesTraining, :]
    Ytrain = Yscrambled[0 : samplesTraining, :]
    Xtest  = Xscrambled[samplesTraining : Xscrambled.shape[0], :]
    Ytest  = Yscrambled[samplesTraining : Xscrambled.shape[0], :]

    return (Xtrain, Ytrain, Xtest, Ytest)

###############################################################################

# Get ArgMax from a Yelement, a single vector of element in matrix Y
def _argmax(Yel):
    _maxArg = float('-inf')
    _maxIdx = - sys.maxsize - 1

    i = 0
    for i in range(len(Yel)):
        if Yel[i] > _maxArg:
            _maxArg = Yel[i]
            _maxIdx = i

    return _maxIdx

###############################################################################

def _calcAccuracy(Y, Ypred):
    if len(Y) != len(Ypred):
        return 0.0

    i = 0
    hits = 0
    total = len(Y)
    for i in range(total):
        # print("Ypred = " + str(Ypred[i]) + " | Ytest = " + str(Ytest[i]))
        # print("Ypred = " + str(argmax(Ypred[i])) + " | Ytest = " + str(argmax(Ytest[i])))
        if _argmax(Ypred[i]) == _argmax(Y[i]):
            hits += 1
    accuracy = hits / total
    
    return accuracy

###############################################################################

# From the label matrix, gets the label vector for a particular class i (one)
# to compare against (vs) all the others together (rest)...
# it seens like is just checking "is class i or not?", then our model test
# how similar to i the current data is... and later we take (choose) the one
# with greater similarity (argmax). Not sure, but it seens heuristic like this
# Y : Label Matrix [N x C]
# i : 0 <= i <= C, stands for the desired class to test (OvR)
def _y(Y, i):
    numberOfSamples = Y.shape[0]

    y = np.empty((numberOfSamples, 1))
    for row in range(numberOfSamples):
        y[row] = _shl(Y[row, i])

    return y

###############################################################################

# element-wise product
def _hadamard_product(u, v):
    return u * v

def _outer_product(u, v):
    return u @ v.T

###############################################################################

# symmetric hard limit (hardlims), used as activation function
def _shl(num):
    return 1 if num >= 0 else -1

###############################################################################

# tangent hyperbolic function, used as activation function
def _tanh(x):
    # return np.tanh(x)
    ex1 = math.exp( x)
    ex2 = math.exp(-x)
    num = ex1 - ex2
    den = ex1 + ex2
    res = num / den
    return res

# apply tanh for a entire vector
def _tanh_vec(vec):
    _vec = vec.copy()

    i = 0
    for i in range(_vec.shape[0]):
        _oldval = _vec[i, 0]
        _newval = _tanh(_oldval)
        _vec[i, 0] = _newval
    return _vec

# derivative of hyperbolic tangent function
def _ddxtanh(x):
    sech = 2.0 / (math.exp(x) + math.exp(-x))
    return sech * sech

# apply ddxtanh for a entire vector
def _ddxtanh_vec(vec):
    _vec = vec.copy()

    i = 0
    for i in range(_vec.shape[0]):
        _oldval = _vec[i, 0]
        _newval = _ddxtanh(_oldval)
        _vec[i, 0] = _newval
    return _vec

# normalised inputs by local normalization
# see also : https://ai.stackexchange.com/questions/5928/
#            how-to-solve-the-problem-of-too-big-activations-when-using-genetic-algorithms-to
def _normalize_samples_local(X):
    for i in range(X.shape[0]):
        x = X[i, :]
        x.shape = (1, X.shape[1])
        norma = math.sqrt(x[0, 0] * x[0, 0] + x[0, 1] * x[0, 1])
        x[0, 0] /= norma
        x[0, 1] /= norma
    return X

def _normalize_samples_minmax(X):
    min_X = X.min()
    max_X = X.max()
    gap_X = max_X - min_X

    _X = X.copy()
    for i in range(_X.shape[0]):
        for j in range(_X.shape[1]):
            xj = _X[i, j]
            xj_norm = (xj - min_X) / gap_X
            xj_norm = 2 * xj_norm - 1
            _X[i, j] = xj_norm
    
    return _X

###############################################################################
