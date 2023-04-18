import numpy as np

class loss:
    def __init__(self):
        pass

    def forward(self,y,yhat) -> float:
        raise NotImplementedError
    
    def backward(self,y, yhat)->np.ndarray:
        raise NotImplementedError 


class mse(loss):
    def forward(self, y, yhat):
        return np.mean((y-yhat)**2)
    
    def backward(self, y, yhat):
        return -2/len(y) * (y - yhat)
#
# class cross_entropy_softmax(loss):
#     def __init__(self,epsilon=1e-12):
#         super().__init__()
#         self.epsilon = epsilon
#
#     def softmax(self,x):
#         exp = np.exp(x)
#         return exp/np.sum(exp)
#
#     def forward(self, y,yhat):
#         softied = self.softmax(yhat)
#         clipped = np.clip(softied, self.epsilon, 1-self.epsilon)
#         return -np.sum(y * np.log(clipped))
#
#     def backward(self, y, yhat) -> np.ndarray:
#         softied = self.softmax(yhat)
#         return softied - y

class cross_entropy_softmax(loss):
    def __init__(self):
        pass
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=0))
        return exp_x / np.sum(exp_x, axis=0)
    
    def forward(self, y, yhat):

        y_pred = self.softmax(yhat)
        loss = -np.sum(y * np.log(y_pred), axis=0)
        return np.mean(loss)
    
    def backward(self, y, yhat):
        y_pred = self.softmax(yhat)
        return y_pred - y
