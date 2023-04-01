# from turtle import forward
import numpy as np
import math
from snn.optimizer import RMSprop, adam, basic, momentum

class Dense:
    def __init__(self, neurons, inpt) -> None:
        self.neurons = neurons
        self.type = "layer"
        self.inpt = inpt
    
    def construct(self, **kwargs):
        return dense(self.neurons, self.inpt, **kwargs)

class Relu:
    def __init__(self) -> None:
        self.type = "activation"
    
    def construct(self):
        return relu()

class Sigmoid:
    def __init__(self) -> None:
        self.type = "activation"
    
    def construct(self):
        return relu()

class Dropout:
    def __init__(self,drop_rate=0.2) -> None:
        self.type = "dropout"
        self.drop_rate= drop_rate

    def construct(self):
        return dropout(self.drop_rate)

class Tanh:
    def __init__(self) -> None:
        self.type = "activation"
    def construct(self):
        return tanh()


class dense:
    def __init__(self,neurons, inpt,optimizer,kaiming=False, xavier=False) -> None:

        if kaiming or xavier:
            init = 1 if xavier else 2
            self.w = (2 * np.random.rand(neurons,inpt) -1 ) * math.sqrt(init/inpt)
        else:
            self.w = np.random.rand(neurons, inpt) - 0.5
        
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
        
        print(self.w_optim)
        self.b = np.zeros((neurons,1))
        self.neurons = neurons
    
    def forward(self,x):
        self.x = x
        return self.w.dot(x) + self.b
    
    def backward(self,otpt_grad,lr, avg):
        dw = np.dot(otpt_grad, self.x.T)
        db = np.sum(otpt_grad, axis=1, keepdims=True)
        dz_prev = np.dot(self.w.T,otpt_grad)
        
        d_w = self.w_optim.step(dw)
        d_b = self.b_optim.step(db)

        self.w -= (lr * d_w * avg)
        self.b -= (lr * d_b * avg)
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


class dropout:
    def __init__(self,drop_rate) -> None:
        self.drop_rate = drop_rate
        self.d_mat = None
    
    def forward(self,x):
        self.d_mat = np.random.rand(*x.shape)
        return np.multiply((self.d_mat > self.drop_rate),x)
    
    def backward(self, otpt_grad, *args, **kwargs):
        return np.multiply(otpt_grad,(self.d_mat/self.drop_rate))


class tanh:
    def __init__(self) -> None:
        pass
    def forward(self, x):
        self.x = x
        return np.tanh(x)
    def backward(self, otpt_grad, *args, **kwargs):
        return np.multiply(otpt_grad, np.cosh(self.x) ** 2)