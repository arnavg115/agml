import numpy as np

class _layer:
    def __init__(self) -> None:
        pass
    def forward(self,x)->np.ndarray:
        raise NotImplementedError
    def backward(self, otpt_grad)->np.ndarray:
        raise NotImplementedError

class Layer:
    def __init__(self) -> None:
        pass
    
    def construct(self)-> _layer:
        raise NotImplementedError