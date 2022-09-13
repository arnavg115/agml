from snn.layers import Dense, softmax
from snn.nn import NN
import numpy as np

from snn.utils import mse, one_hot, relu
from snn.utils.activations import sigmoid
from snn.utils.losses import cross_ent
import pandas as pd

df = pd.read_csv("mnist_train_small.csv")
x = df.to_numpy()[:,1:].T/255
y = one_hot(df.to_numpy()[:,0])
# print(x.T[0:2].T.shape)
# print(np.argmax(y[0]))

# print(one_hot(y))
k = NN([Dense(784,40),sigmoid(),Dense(40,10), sigmoid()],loss=mse,lr=0.01)

k.train(x, y,200)

print(np.argmax(k.predict(x.T[0:10].T),axis=0))
print(np.argmax(y[0:10],axis = 0))

# k.forward()

# # print(type(k.loss))

# # ot = k.forward(np.ones((10,1000)),np.ones((2,1000)))
# k.backward()


# print(1/y.shape[0] * np.sum((yhat-y)))

# print(k.forward(np.ones(10,1000)).T,label=np.random.randn((2,1000)))

# print(relu)

# rand = np.random.rand(2,5)

# rand2 = np.random.rand(2,5)

# print(np.argmax(rand,axis=1) == np.argmax(rand2, axis=1))

