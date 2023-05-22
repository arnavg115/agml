import numpy as np

class loss:
    def __init__(self):
        pass

    def forward(self,y,yhat) -> float:
        raise NotImplementedError
    
    def backward(self,y, yhat)->np.ndarray:
        raise NotImplementedError
    
    def backward_prop(self):
        raise NotImplementedError


class mse(loss):
    def forward(self, y, yhat):
        self.y = y
        self.yhat = yhat
        return np.mean((y-yhat)**2)
    
    def backward(self, y, yhat):
        return -2/len(y) * (y - yhat)
    
    def backward_prop(self):
        return self.backward(self.y, self.yhat)
        

class cross_entropy_softmax(loss):
    def __init__(self):
        pass
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, y, yhat, axis = 0):
        self.y = y
        self.yhat = yhat
        y_pred = self.softmax(yhat)
        loss = -np.sum(y * np.log(y_pred), axis=axis)
        return np.mean(loss)
    
    def backward(self, y, yhat):
        y_pred = self.softmax(yhat)
        return y_pred - y
    
    def backward_prop(self):
        return self.backward(self.y, self.yhat)
