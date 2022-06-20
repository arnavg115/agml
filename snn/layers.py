import numpy as np

from snn.utils.activations import relu


class Dense:
    def __init__(self, inpt_sh, neurons,act:relu) -> None:
        self.w:np.array = np.random.randn(neurons, inpt_sh)
        self.b = np.random.randn(neurons, 1)
        self.act =  act

    
    def forward(self,inpt):
        self.inpt = inpt
        self.intermed = np.dot(self.w, inpt) + self.b
        print(self.intermed.shape)
        return self.act.run(self.intermed)

    # todo: fix backprop error
    def backward(self, otpt_grad):
        print(otpt_grad.shape)
        act_d = self.act.der(self.intermed) * otpt_grad
        w_grad = np.dot(act_d,self.inpt.T)
        b_grad = act_d
        return act_d



class softmax:
    def __init__(self) -> None:
        pass
    def forward(self, inpt):
        self.otpt = np.exp(inpt)/np.sum(np.exp(inpt))
        return self.otpt
    
    def backward(self,y_labels):
        return  self.otpt - y_labels
    