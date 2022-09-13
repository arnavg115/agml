import numpy as np



class relu:
    tp = "act"
    @staticmethod
    def forward(x):
        return np.maximum(0,x)
    @staticmethod
    def backward(x,lr):
        return (relu.forward(x) != 0).astype(int)



class sigmoid:

    tp = "act"
    @staticmethod
    def forward(x):
        return 1/(1+np.exp(-x))

    @staticmethod
    def backward(x,lr):
        forw = sigmoid.forward(x)
        return forw * (1-forw)


# relu = lambda x: np.maximum(0,x)
# relu_der = lambda x: 0 if relu(x) == 0 else 1
