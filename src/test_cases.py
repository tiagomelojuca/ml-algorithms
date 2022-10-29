###############################################################################
# THIRD-PARTY CODE
# DataSet and Algorithm used for tests by Paulo Cirillo Souza Barbosa
###############################################################################
import os
import json
import numpy as np
import matplotlib.pyplot as plt
###############################################################################

def DUMMYREG_getData():
    X = np.array([ 480, 500, 380, 1100, 1100, 230, 490, 250, 300, 510 ])
    X.shape = (len(X), 1)
    Y = np.array([ 180, 150, 170,  350,  460,  60, 240,  90, 110, 250 ])
    Y.shape = (len(Y), 1)

    return (X, Y)

###############################################################################

def DUMMYCLS_getData():
    X = np.array([[1, 1], [0, 1], [0, 2], [1, 0], [2, 2], [4, 1.5], [1.5, 6], [3, 5], [3, 3], [6, 4]])
    Y = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1])
    Y.shape = (len(Y), 1)

    return (X, Y)

###############################################################################

def DUMMYCLS_showData():
    X, Y = DUMMYCLS_getData()
    plt.scatter(X[0 :  5, 0], X[0 :  5, 1], color = "blue")
    plt.scatter(X[5 : 10, 0], X[5 : 10, 1], color = "red" )
    plt.xlim((-1, 7))
    plt.ylim((-1, 7))
    plt.show()

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

def DUMMYMULTICLS_getData():
    X = np.array([
        [2.5, 0.0], # DUMMY SANPLE FOR CLASS 1
        [2.3, 0.7], # DUMMY SANPLE FOR CLASS 1
        [2.1, 0.3], # DUMMY SANPLE FOR CLASS 1
        [2.7, 0.3], # DUMMY SANPLE FOR CLASS 1
        [2.9, 0.1], # DUMMY SANPLE FOR CLASS 1
        [2.2, 0.5], # DUMMY SANPLE FOR CLASS 1
        [2.7, 0.2], # DUMMY SANPLE FOR CLASS 1
        [2.3, 0.1], # DUMMY SANPLE FOR CLASS 1
        [2.5, 0.6], # DUMMY SANPLE FOR CLASS 1
        [2.2, 0.9], # DUMMY SANPLE FOR CLASS 1

        [0.0, 3.0], # DUMMY SANPLE FOR CLASS 2
        [0.2, 3.7], # DUMMY SANPLE FOR CLASS 2
        [0.9, 3.1], # DUMMY SANPLE FOR CLASS 2
        [0.3, 3.3], # DUMMY SANPLE FOR CLASS 2
        [0.2, 3.5], # DUMMY SANPLE FOR CLASS 2
        [0.5, 3.3], # DUMMY SANPLE FOR CLASS 2
        [0.4, 3.9], # DUMMY SANPLE FOR CLASS 2
        [0.3, 3.7], # DUMMY SANPLE FOR CLASS 2
        [0.9, 3.6], # DUMMY SANPLE FOR CLASS 2
        [0.1, 3.4], # DUMMY SANPLE FOR CLASS 2

        [6.5, 1.5], # DUMMY SANPLE FOR CLASS 3
        [6.2, 1.3], # DUMMY SANPLE FOR CLASS 3
        [6.1, 1.5], # DUMMY SANPLE FOR CLASS 3
        [6.0, 1.4], # DUMMY SANPLE FOR CLASS 3
        [6.6, 1.2], # DUMMY SANPLE FOR CLASS 3
        [6.0, 1.7], # DUMMY SANPLE FOR CLASS 3
        [6.8, 1.9], # DUMMY SANPLE FOR CLASS 3
        [6.3, 1.3], # DUMMY SANPLE FOR CLASS 3
        [6.1, 1.4], # DUMMY SANPLE FOR CLASS 3
        [6.4, 1.6], # DUMMY SANPLE FOR CLASS 3

        [3.4, 6.0], # DUMMY SANPLE FOR CLASS 4
        [3.0, 6.2], # DUMMY SANPLE FOR CLASS 4
        [3.7, 6.0], # DUMMY SANPLE FOR CLASS 4
        [3.4, 6.6], # DUMMY SANPLE FOR CLASS 4
        [3.7, 6.1], # DUMMY SANPLE FOR CLASS 4
        [3.2, 6.7], # DUMMY SANPLE FOR CLASS 4
        [3.0, 6.1], # DUMMY SANPLE FOR CLASS 4
        [3.1, 6.3], # DUMMY SANPLE FOR CLASS 4
        [3.0, 6.8], # DUMMY SANPLE FOR CLASS 4
        [3.5, 6.7], # DUMMY SANPLE FOR CLASS 4

        [6.1, 4.5], # DUMMY SANPLE FOR CLASS 5
        [6.5, 4.6], # DUMMY SANPLE FOR CLASS 5
        [6.2, 4.5], # DUMMY SANPLE FOR CLASS 5
        [6.2, 4.8], # DUMMY SANPLE FOR CLASS 5
        [6.6, 4.5], # DUMMY SANPLE FOR CLASS 5
        [6.0, 4.8], # DUMMY SANPLE FOR CLASS 5
        [6.6, 4.5], # DUMMY SANPLE FOR CLASS 5
        [6.8, 4.9], # DUMMY SANPLE FOR CLASS 5
        [6.3, 4.8], # DUMMY SANPLE FOR CLASS 5
        [6.1, 4.9], # DUMMY SANPLE FOR CLASS 5
    ])

    Y = np.array([
        [ 1, -1, -1, -1, -1],
        [ 1, -1, -1, -1, -1],
        [ 1, -1, -1, -1, -1],
        [ 1, -1, -1, -1, -1],
        [ 1, -1, -1, -1, -1],
        [ 1, -1, -1, -1, -1],
        [ 1, -1, -1, -1, -1],
        [ 1, -1, -1, -1, -1],
        [ 1, -1, -1, -1, -1],
        [ 1, -1, -1, -1, -1],

        [-1,  1, -1, -1, -1],
        [-1,  1, -1, -1, -1],
        [-1,  1, -1, -1, -1],
        [-1,  1, -1, -1, -1],
        [-1,  1, -1, -1, -1],
        [-1,  1, -1, -1, -1],
        [-1,  1, -1, -1, -1],
        [-1,  1, -1, -1, -1],
        [-1,  1, -1, -1, -1],
        [-1,  1, -1, -1, -1],

        [-1, -1,  1, -1, -1],
        [-1, -1,  1, -1, -1],
        [-1, -1,  1, -1, -1],
        [-1, -1,  1, -1, -1],
        [-1, -1,  1, -1, -1],
        [-1, -1,  1, -1, -1],
        [-1, -1,  1, -1, -1],
        [-1, -1,  1, -1, -1],
        [-1, -1,  1, -1, -1],
        [-1, -1,  1, -1, -1],

        [-1, -1, -1,  1, -1],
        [-1, -1, -1,  1, -1],
        [-1, -1, -1,  1, -1],
        [-1, -1, -1,  1, -1],
        [-1, -1, -1,  1, -1],
        [-1, -1, -1,  1, -1],
        [-1, -1, -1,  1, -1],
        [-1, -1, -1,  1, -1],
        [-1, -1, -1,  1, -1],
        [-1, -1, -1,  1, -1],

        [-1, -1, -1, -1,  1],
        [-1, -1, -1, -1,  1],
        [-1, -1, -1, -1,  1],
        [-1, -1, -1, -1,  1],
        [-1, -1, -1, -1,  1],
        [-1, -1, -1, -1,  1],
        [-1, -1, -1, -1,  1],
        [-1, -1, -1, -1,  1],
        [-1, -1, -1, -1,  1],
        [-1, -1, -1, -1,  1],
    ])

    return (X, Y)

###############################################################################

def DUMMYMULTICLS_showData():
    X, Y = DUMMYMULTICLS_getData()
    plt.scatter(X[ 0 : 10, 0], X[ 0 : 10, 1], color = "blue" )
    plt.scatter(X[10 : 20, 0], X[10 : 20, 1], color = "red"  )
    plt.scatter(X[20 : 30, 0], X[20 : 30, 1], color = "green")
    plt.scatter(X[30 : 40, 0], X[30 : 40, 1], color = "pink")
    plt.scatter(X[40 : 50, 0], X[40 : 50, 1], color = "orange")
    plt.xlim((-1, 7))
    plt.ylim((-1, 7))
    plt.show()

###############################################################################
