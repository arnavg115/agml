import pandas as pd
import numpy as np
from snn.layer import dense, relu, softmax
from snn.loss import cross_ent
from snn.nn import NN


df = pd.read_csv("train.csv")
data = df.to_numpy()
x = np.reshape(data[:,1:],(42000,784,1)) / 255
y = np.reshape(data[:,0],(42000))
x_train = x[:10000]
y_train = y[:10000]
m, = y_train.shape
# print(np.identity(y)[0:0+1])
one_hot = np.zeros((y_train.shape[0],10))
one_hot[np.arange(m),y_train] = 1
one_hot = np.reshape(one_hot, (y_train.shape[0],10,1))
# print(one_hot.shape)
nn = NN([dense(20,784),relu(),dense(10,20),softmax()],loss=cross_ent())

nn.train(10,x_train,one_hot)

NN.save("model.pkl",nn)
