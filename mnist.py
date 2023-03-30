import numpy as np
from snn.layer import dense, relu, sigmoid
from snn.loss import mse
from snn.nn import NN
from snn.utils import one_hot_func
import csv
import os
from snn.vis import display_mnist_image

if "train.csv" not in os.listdir():
    import urllib.request
    print("Downloading data")
    url = 'https://github.com/arnavg115/snn/releases/download/DATA/train.csv'
    urllib.request.urlretrieve(url, 'train.csv')

out = []

with open('train.csv') as csvfile:
    reader = csv.reader(csvfile)
    out = list(reader)

data = np.array(out,dtype=int)
x = np.reshape(data[:,1:],(data.shape[0],784)) / 255

y = np.reshape(data[:,0],(data.shape[0]))
one_hot = one_hot_func(y)

if "model.pkl" in os.listdir():
    print("Loading model from detected saved checkpoint")
    nn = NN.load("model.pkl")
else:
    nn = NN([dense(128,784, kaiming=True), relu(), dense(10,128, kaiming=True)], loss=mse(), lr=0.01)
    nn.train(200, x[10000:], one_hot[10000:], batch_size=500)
    NN.save("model.pkl", nn)



while True:
    ins = input("Enter index of image to see and predict (0 - 9999), or enter q to quit: ")
    if ins == "q":
        break
    ind = int(ins)
    display_mnist_image(data[ind,1:].reshape((28,28)))
    print(np.argmax(nn.forward(x[ind:ind+1])))