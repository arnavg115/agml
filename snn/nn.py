# from turtle import forward
import numpy as np
from tqdm import tqdm
import pickle


class NN:
    def __init__(self, layers,loss, lr = 0.01,) -> None:
        self.layers = layers
        self.lr = lr
        self.loss = loss
    
    def forward(self, x):
        st = x
        
        for layer in self.layers:
            st = layer.forward(st)
            # print(st.shape)
        
        # self.ot = st
        return st

    def backward(self, y, yhat):
        grad = self.loss.backward(y,yhat)
        # print(y.shape)
        for layer in reversed(self.layers):
            grad = layer.backward(grad,self.lr)
        
        los = self.loss.forward(y, yhat)
        return los

    def train(self,epochs, X,Y):
        tq = tqdm(range(epochs))
        print(Y.shape)
        
        for epoch in tq:
            losss = 0
            accu = 0
            for x,y in zip(X,Y):
                yhat = self.forward(x)
                accu += np.argmax(yhat) == np.argmax(y)
                los = self.backward(y,yhat)
                losss += los
            tq.set_description_str(f"LOSS:{losss/len(Y)}, ACCU:{accu/len(Y)}")
    
    
    def validate(self,x_val,y_val):
        return np.sum(np.argmax(self.forward(x_val.T),axis = 0) == y_val)/y_val.shape[0]
    
    @staticmethod
    def save(filename:str,nn):
        file = open(filename,"wb")
        pickle.dump(nn,file)
    
    @staticmethod
    def load(filename:str):
        file = open(filename, "rb")
        return pickle.load(file)
            