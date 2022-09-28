# from turtle import forward
import numpy as np


class dense:
    def __init__(self,neurons, inpt) -> None:
        self.w = np.random.rand(neurons, inpt) - 0.5
        self.b = np.random.rand(neurons, 1) - 0.5
    
    def forward(self,x):
        self.x = x
        return self.w.dot(x) + self.b
    
    def backward(self,otpt_grad,lr):
        dw = np.dot(otpt_grad, self.x.T)
        db = otpt_grad
        dz_prev = np.dot(self.w.T,otpt_grad)
        
        self.w -= lr * dw
        self.b -= lr * db
        return dz_prev

class relu:
    def __init__(self) -> None:
        pass
    
    def forward(self, x):
        self.x = x
        return np.maximum(x, 0)

    def backward(self, otpt_grad, *args):
        return np.multiply(otpt_grad, self.x > 0)

class sigmoid:
    def __init__(self) -> None:
        pass
    
    def forward(self, x):
        self.x = x
        return 1 / (1+ np.exp(-x))
    
    def backward(self, otpt_grad, *args):

        fwd = self.forward(self.x)
        intermed = fwd * (1-fwd)
        
        return np.multiply(otpt_grad, intermed)
        

class softmax:
    def __init__(self) -> None:
        pass
    
    def forward(self, x):
        ex = np.exp(x)
        return ex / np.sum(ex)
    
    def backward(self, otpt_grad, *args):
        return otpt_grad