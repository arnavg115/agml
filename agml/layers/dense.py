from .layer import _layer, Layer
import numpy as np
from ..optimizer import basic, momentum, RMSprop, adam
import math


class Dense(Layer):
    def __init__(self, neurons, inpt) -> None:
        self.neurons = neurons
        self.type = "layer"
        self.inpt = inpt
    
    def construct(self, **kwargs):
        return dense(self.neurons, self.inpt, **kwargs)

class dense(_layer):
    def __init__(self,neurons, inpt,optimizer,kaiming=False, xavier=False) -> None:
        super().__init__()
        if kaiming or xavier:
            init = 1 if xavier else 2
            self.w = (2 * np.random.rand(inpt,neurons) -1 ) * math.sqrt(init/inpt)
        else:
            self.w = np.random.rand(inpt, neurons) - 0.5
        
        self.w_optim = basic()
        self.b_optim = basic() 
        
        if optimizer == "momentum":
            self.w_optim = momentum()
            self.b_optim = momentum()
        elif optimizer == "rmsprop":
            self.w_optim = RMSprop()
            self.b_optim = RMSprop()
        elif optimizer == "adam":
            self.w_optim = adam()
            self.b_optim = adam()
        
        self.b = np.zeros((1,neurons))
        self.neurons = neurons
    
    def forward(self,x):
        self.x = x 
        return self.x.dot(self.w) + self.b
    
    def backward(self,otpt_grad,lr, avg):
        # dw = np.dot(otpt_grad, self.x.T)
        dw = np.dot(self.x.T, otpt_grad)
        db = np.sum(otpt_grad, axis=0, keepdims=True)
        # dz_prev = np.dot(self.w.T,otpt_grad)
        dz_prev = np.dot(otpt_grad, self.w.T)
        
        d_w = self.w_optim.step(dw)
        d_b = self.b_optim.step(db)
        
        self.w -= (lr * d_w * avg)
        self.b -= (lr * d_b * avg)
        return dz_prev
