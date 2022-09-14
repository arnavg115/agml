import numpy as np
from .utils.activations import relu


class Dense:

    def __init__(self, inpt_sh, neurons) -> None:
        self.w:np.array = np.random.rand(neurons, inpt_sh) - 0.5
        self.b = np.random.rand(neurons, 1) - 0.5

    def forward(self,inpt):
        self.inpt = inpt
        return np.dot(self.w, inpt) + self.b

    # todo: fix backprop error
    def grads(self, otpt_grad):
        # print(otpt_grad.shape)
        w_grad = np.dot(otpt_grad,self.inpt.T)
        b_grad = np.sum(otpt_grad)
        # print(self.w.T.shape)
        z_grad = self.w.T.dot(otpt_grad)
        # print(z_grad.shape)
        # return update_params(otpt_grad,lr)
        return w_grad, b_grad, z_grad
    
    def backward(self, otpt_grad,lr):
        # print(self.b.shape)

        w_grad, b_grad, z_grad = self.grads(otpt_grad)
        self.w = self.w - (lr * w_grad)
        self.b = self.b - (lr * b_grad)
        
        return z_grad 


        



class softmax:
    def __init__(self) -> None:
        pass
    def forward(self, inpt):
        self.otpt = np.exp(inpt)/np.sum(np.exp(inpt))
        return self.otpt
    
    def backward(self,y_labels,lr):
        return  self.otpt - y_labels
    