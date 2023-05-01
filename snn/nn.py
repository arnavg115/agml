from typing import List
import numpy as np
from tqdm import tqdm
import pickle
from .layers import Layer
from .layers.layer import _layer, Layer
from .loss import loss
from .utils import accuracy


class NN:
    def __init__(self, layers:List[Layer],loss:loss, lr = 0.01,optimizer = None, kaiming=False, xavier=False) -> None:
        self.layers:List[_layer] = []

        for layer in layers:
            if layer.type == "activation" or layer.type=="dropout":
                self.layers.append(layer.construct())
            else:
                self.layers.append(layer.construct(kaiming=kaiming, xavier=xavier,optimizer=optimizer ))
        self.lr = lr
        self.loss = loss
    
    def forward(self, x):
        st = x
        for layer in self.layers:
            st = layer.forward(st)
        return st

    def backward(self, y_inp, yhat):
        y = y_inp

        grad = self.loss.backward(y,yhat)

        # print(y.shape)
        for layer in reversed(self.layers):
            grad = layer.backward(grad,self.lr, avg = 1/y.shape[1])
        
        los = self.loss.forward(y, yhat)
        return los

    def train(self,epochs, X,Y, batch_size=0):
        tq = tqdm(range(epochs))
        if batch_size !=0:
            numb = X.shape[0] // batch_size

        for epoch in tq:
            if batch_size == 0:
                yhat = self.forward(X)
                los = self.backward(Y, yhat)
                tq.set_description_str(f"LOSS:{los}, ACCU:{accuracy(yhat, Y)}")
            else:
                loss = 0
                accu = 0
                for batch in range(numb):
                    x_tr = X[batch_size*batch: batch_size*(batch+1)]
                    y_tr = Y[batch_size*batch: batch_size*(batch+1)]

                    yhat = self.forward(x_tr)
                    accu+= np.sum(np.argmax(yhat, axis=1) == np.argmax(y_tr, axis=1))
                    loss+=self.backward(y_tr, yhat)
                
                tq.set_description_str(f"AVG BATCH LOSS:{loss/numb}, BATCH ACCURACY:{accu/(numb * y_tr.shape[0])}")
                  
    def predict(self, x):
        return self.forward(x[np.newaxis,:])

    def validate(self,x_val,y_val):
        return accuracy(self.forward(x_val), y_val)
    
    def forward_layer(self, x, ind:int):
        return self.layers[ind].forward(x)
    
    @staticmethod
    def save(filename:str,nn):
        file = open(filename,"wb")
        pickle.dump(nn,file)
    
    @staticmethod
    def load(filename:str):
        file = open(filename, "rb")
        return pickle.load(file)
            
