import numpy as np
from snn.layer import Dense, Relu
from snn.loss import cross_entropy_softmax, mse
from snn.nn import NN
from snn.utils import one_hot_func
import csv
import os
from snn.vis import display_mnist_image
import urllib.request


download_data = False
download_weights = False


    
if download_data:
  print("Downloading MNIST Data")
  url = "https://github.com/arnavg115/snn/releases/download/DATA/train.csv"
  urllib.request.urlretrieve(url, 'train.csv')
if download_weights:
  print("Downloading pretrained model weights")
  url= "https://github.com/arnavg115/snn/releases/download/DATA/model.pkl"
  filename = 'model.pkl'
  with urllib.request.urlopen(url) as response, open(filename, 'wb') as out_file:
      data = response.read()  # read the file contents in memory
      out_file.write(data) 


out = []
print("Opening Data")
with open('train.csv') as csvfile:
    reader = csv.reader(csvfile)
    out = list(reader)

data = np.array(out,dtype=int)
x = data[:,1:] / 255

y = data[:,0]
one_hot = one_hot_func(y,10)

if "model.pkl" in os.listdir():
    print("Loading model from detected saved checkpoint")
    nn = NN.load("model.pkl")
else:
    nn = NN([Dense(128,784), Relu(), Dense(10,128)], loss=cross_entropy_softmax(), lr=0.01, kaiming=True, optimizer="adam")
    nn.train(50, x, one_hot, batch_size=200)
    # NN.save("model.pkl", nn)
#



print("Starting inference")
while True:
    ins = input("Enter index of image to see and predict (0 - 9999), or enter q to quit: ")
    if ins == "q":
        break
    ind = int(ins)
    display_mnist_image(x[ind].reshape((28,28)))
    print(f"Predicted: {np.argmax(nn.predict(x[ind]))}")
    print(f"Actual: {y[ind]}")
