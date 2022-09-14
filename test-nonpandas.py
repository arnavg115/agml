import csv
import numpy as np
from snn.layers import Dense, softmax
from snn.nn import NN
from snn.utils import one_hot
from snn.utils.activations import relu, sigmoid
from snn.utils.losses import cross_ent, mse

i =[]


with open("mnist_train_small.csv") as file:
    reader = csv.reader(file)
    for line in reader:
        i.append(line)

data = np.array(i,dtype=np.float)
x = data[:100,1:].T / 255 
y = one_hot(data[:,0].astype(np.int))[:,:100]


nn = NN(layers=[Dense(784,40),sigmoid(), Dense(40,10), sigmoid()],loss=mse, lr=0.5)

nn.train(x,y, epochs=5000)