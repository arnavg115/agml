import csv
import numpy as np
from snn.layers import Dense, softmax
from snn.nn import NN
from snn.utils import one_hot
from snn.utils.activations import relu, sigmoid
from snn.utils.losses import cross_ent, mse

i =[]


with open("data.csv") as file:
    reader = csv.reader(file)
    for line in reader:
        i.append(line)


data = np.array(i,dtype=np.float)
x = data[:,0:2].T
x = x/np.max(x)
y = one_hot(data[:,2].astype(np.int))
print(x.shape)


nn = NN(layers=[Dense(2,2), sigmoid()],loss=mse, lr=0.5)
#
nn.train(x,y, epochs=5000)