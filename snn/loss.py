import numpy as np

class mse:
    def __init__(self) -> None:
        pass

    def forward(self, y, yhat):
        return np.sum(np.power((y-yhat),2))
    
    def backward(self, y, yhat):
        return 2/len(y) * (yhat - y)
    

class cross_ent:
    def __init__(self) -> None:
        pass
    
    def forward(self, y,yhat):
        return -1 * np.sum(y * np.log10(yhat))
    
    def backward(self, y, yhat):
        return yhat - y

