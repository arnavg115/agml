from snn.layers import Dense, softmax
from snn.nn import NN
import numpy as np

from snn.utils import mse, relu
from snn.utils.losses import cross_ent

k = NN([Dense(10,3,relu),Dense(3,2,relu),softmax()],loss=cross_ent)

print(k.train(np.ones((10,1000)),np.ones((2,1000))))

# print(type(k.loss))

# ot = k.forward(np.ones((10,1000)),np.ones((2,1000)))
k.backward()


# print(1/y.shape[0] * np.sum((yhat-y)))

# print(k.forward(np.ones(10,1000)).T,label=np.random.randn((2,1000)))

# print(relu)

