# from turtle import forward
import numpy as np
import math

class dense:
    def __init__(self,neurons, inpt, kaiming=False, xavier=False) -> None:

        if kaiming or xavier:
            init = 1 if xavier else 2
            self.w = (2 * np.random.rand(neurons, inpt) -1 ) * math.sqrt(init/inpt)
        else:
            self.w = np.random.rand(neurons, inpt) - 0.5
        self.b = np.zeros((neurons,1))
        self.neurons = neurons
    
    def forward(self,x):
        self.x = x
        return self.w.dot(x) + self.b
    
    def backward(self,otpt_grad,lr, avg):
        dw = np.dot(otpt_grad, self.x.T)
        db = otpt_grad
        dz_prev = np.dot(self.w.T,otpt_grad)
        
        self.w -= (lr * dw * avg)
        self.b -= (lr * np.sum(db, axis=1,keepdims=True) * avg)
        return dz_prev


class relu:
    def __init__(self) -> None:
        pass
    
    def forward(self, x):
        self.x = x
        return np.maximum(x, 0)

    def backward(self, otpt_grad, *args, **kwargs):
        return np.multiply(otpt_grad, self.x > 0)

class sigmoid:
    def __init__(self) -> None:
        pass
    
    def forward(self, x):
        self.x = x
        return 1 / (1+ np.exp(-x))
    
    def backward(self, otpt_grad, *args, **kwargs):

        fwd = self.forward(self.x)
        intermed = fwd * (1-fwd)
        
        return np.multiply(otpt_grad, intermed)
        