import numpy as np


def one_hot_func(array, size):
    m = len(array)
    one_hot = np.zeros((m,size))
    one_hot[np.arange(m),array] = 1
    return one_hot

def accuracy(a, b):
    return sum(np.argmax(a, axis=1) == np.argmax(b, axis=1))/a.shape[0]