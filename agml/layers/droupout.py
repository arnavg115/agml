from .layer import _layer, Layer
import numpy as np

class Dropout(Layer):
    def __init__(self,drop_rate=0.2) -> None:
        self.type = "dropout"
        self.drop_rate= drop_rate

    def construct(self):
        return dropout(self.drop_rate)


class dropout(_layer):
    def __init__(self,drop_rate) -> None:
        self.drop_rate = drop_rate
        self.d_mat = None
    
    def forward(self,x):
        self.d_mat = np.random.rand(*x.shape)
        return np.multiply((self.d_mat > self.drop_rate),x)
    
    def backward(self, otpt_grad, *args, **kwargs):
        return np.multiply(otpt_grad,(self.d_mat/self.drop_rate))