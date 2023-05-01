# import imp
from snn.layers import dense, relu, sigmoid
from snn.loss import mse
from snn.nn import NN
import numpy as np
import pickle


X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

nn = NN([dense(3,2), relu(),dense(1,3),sigmoid()],loss=mse())

nn.train(10000,X,Y)

# print(np.size(Y))
for x in X:
    print(nn.forward(x))

NN.save("model.pkl",nn)

