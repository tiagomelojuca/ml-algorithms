###############################################################################
# THIRD-PARTY CODE
# DataSet and Algorithm used for tests by Paulo Cirillo Souza Barbosa
###############################################################################
import os
import json
import numpy as np
import matplotlib.pyplot as plt
###############################################################################

def DUMMY_getData():
    X = np.array([ 480, 500, 380, 1100, 1100, 230, 490, 250, 300, 510 ])
    X.shape = (len(X), 1)
    Y = np.array([ 180, 150, 170,  350,  460,  60, 240,  90, 110, 250 ])
    Y.shape = (len(Y), 1)

    return (X, Y)

###############################################################################

def D6V6PADRAO_getData():
    f = open(os.path.dirname(__file__) + "/../datasets/D6V6PADRAO.json")
    c = 5 # 'Neutro', 'Sorriso', 'Aberto', 'Surpreso', 'Grumpy'
    p = 2

    data = json.load(f)
    X = np.empty((2, 0))
    Y = np.empty((c, 0))

    for j in data:
        it = 0
        data1 = data[j]
        for i in data1:
            # X
            aux1 = np.array(data1[i][0])
            aux1.shape = (1, len(aux1))
            aux2 = np.array(data1[i][1])
            aux2.shape = (1, len(aux2))
            seed = np.random.permutation(aux2.shape[1])
            aux1 = aux1[:, seed[0 : 50]]
            aux2 = aux2[:, seed[0 : 50]]
            aux1 = aux1[:, :]
            aux2 = aux2[:, :]
            X = np.concatenate((X, np.concatenate((aux1, aux2), axis = 0)), axis = 1)

            # Y
            y = -np.ones((c, aux2.shape[1]))
            y[it, :] = 1

            Y = np.concatenate((Y, y), axis = 1)
            it += 1
    
    X = X.T
    Y = Y.T

    return (X, Y)

###############################################################################

def D6V6PADRAO_showData():
    X, Y = D6V6PADRAO_getData()

    i = 0
    while i < X.shape[0]:
        plt.scatter(X[0   + i : 50  + i, 0], X[0   + i : 50  + i, 1], color='red'   )
        plt.scatter(X[50  + i : 100 + i, 0], X[50  + i : 100 + i, 1], color='orange')
        plt.scatter(X[100 + i : 150 + i, 0], X[100 + i : 150 + i, 1], color='black' )
        plt.scatter(X[150 + i : 200 + i, 0], X[150 + i : 200 + i, 1], color='gray'  )
        plt.scatter(X[200 + i : 250 + i, 0], X[200 + i : 250 + i, 1], color='purple')
        i = i + 250
    plt.show()

###############################################################################
