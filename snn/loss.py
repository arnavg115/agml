import numpy as np

class mse:
    def __init__(self) -> None:
        pass

    def forward(self, y, yhat):
        return np.mean((y-yhat)**2)
    
    def backward(self, y, yhat):
        return -2/len(y) * (y - yhat)


