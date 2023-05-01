from .layer import Layer, _layer
import numpy as np

class Tanh(Layer):
    def __init__(self) -> None:
        self.type = "activation"
    def construct(self):
        return tanh()

class Leaky_ReLU(Layer):
    def __init__(self,alpha=0.01) -> None:
        super().__init__()
        self.type = "activation"
        self.alpha = alpha

    def construct(self) -> _layer:
        return leaky_relu(self.alpha)

class Relu(Layer):
    def __init__(self) -> None:
        self.type = "activation"
    
    def construct(self):
        return relu()

class Sigmoid(Layer):
    def __init__(self) -> None:
        self.type = "activation"
    
    def construct(self):
        return relu()



class relu(_layer):
    def __init__(self) -> None:
        pass
    
    def forward(self, x):
        self.x = x
        return np.maximum(x, 0)

    def backward(self, otpt_grad, *args, **kwargs):
        return np.multiply(otpt_grad, self.x > 0)

class sigmoid(_layer):
    def __init__(self) -> None:
        pass
    
    def forward(self, x):
        self.x = x
        return 1 / (1+ np.exp(-x))
    
    def backward(self, otpt_grad, *args, **kwargs):

        fwd = self.forward(self.x)
        intermed = fwd * (1-fwd)
        
        return np.multiply(otpt_grad, intermed)

class tanh(_layer):
    def __init__(self) -> None:
        pass
    def forward(self, x):
        self.x = x
        return np.tanh(x)
    def backward(self, otpt_grad, *args, **kwargs):
        return np.multiply(otpt_grad, 1/np.cosh(self.x) ** 2)
    

class leaky_relu(_layer):
    def __init__(self,alpha=0.01) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x) -> np.ndarray:
        self.x = x
        return np.maximum(x,self.alpha*x)
    def backward(self, otpt_grad, *args, **kwargs):
        grad = np.ones_like(self.x)
        grad[self.x<0] = self.alpha
        return np.multiply(otpt_grad, grad)