from .layers import softmax
from .utils import accuracy
from .utils.losses import cross_ent
import numpy as np

class NN:
    def __init__(self, layers, loss, lr = 1E-5) -> None:
        self.layers = layers
        self.loss = loss
        self.lr  = lr

    def predict(self,x):
        i  = x
        for layer in self.layers:
            i = layer.forward(i)
        return i
    def train(self,x,y,epochs):
        self.x = x
        self.y = y
        for i in range(epochs):
            j, loss, accu = self.forward()
            
            self.backward()

        return i, loss, accu

    def forward(self):
        i = self.x
        for layer in self.layers:

            i = layer.forward(i)
        print(np.shape(i))

        self.yhat = i
        loss = self.loss.run(y=self.y, yhat = i)
        accu = accuracy(y=self.y, yhat = i)
        print(accu)
        return i, loss, accu

    # todo: finish backprop for entire nn
    def backward(self):
        grad = []

        if not self.loss is cross_ent:
            grad = self.loss.der(self.y, self.yhat)
        lrs = self.layers[::-1]
        for layer in lrs:
            grad = layer.backward(grad,self.lr)
            # if type(layer) is softmax:
            #     grad = layer.backward(self.y)
            # else:
            #     grad = layer.update_params(grad, self.lr)
                # print(grad.shape)
            # print(grad.shape)


        




    