from .layers import softmax
from .utils import accuracy
from .utils.losses import cross_ent
import numpy as np
from tqdm import tqdm

class NN:
    def __init__(self, layers, loss, lr = 1E-5, batch_size = 64) -> None:
        self.layers = layers
        self.loss = loss
        self.lr  = lr
        self.batch_size = 64

    def predict(self,x):
        i  = x
        for layer in self.layers:
            i = layer.forward(i)
        return i
        
    def train(self,x,y,epochs):
        self.x = x
        self.y = y
        tq = tqdm(range(epochs))
        for i in tq:
            loss, accu = self.forward(x)
            tq.set_description_str(f"Loss: {loss}, Accuracy: {accu}")
            
            self.backward()

        return i, loss, accu

    def forward(self,x):
        self.yhat = self.predict(x)
        loss = self.loss.run(y=self.y, yhat = self.yhat)
        accu = accuracy(y=self.y, yhat = self.yhat)
        return loss, accu

    # todo: finish backprop for entire nn
    def backward(self):
        grad = self.y

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


        




    