import numpy as np
from .losses import *
from .activations import *

# https://developers.google.com/machine-learning/crash-course/classification/accuracy
def accuracy(yhat, y):
    truepreds = np.argmax(y,axis=1) == np.argmax(yhat, axis=1)
    return sum(truepreds) / len(truepreds)


def one_hot(y):
    enc = np.zeros((len(np.unique(y)),y.shape[0]))
    # print(enc.shape)
    for i,val in enumerate(y):
      enc[val][i] = 1
    return enc