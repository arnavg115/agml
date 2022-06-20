import numpy as np



class relu:
    name = "relu"

    @staticmethod
    def run(x):
        return np.maximum(0,x)
    @staticmethod
    def der(x):
        return (relu.run(x)).astype(int)



class sigmoid:
    name = "sig"

    @staticmethod
    def run(x):
        return 1/(1+np.exp(-x))

    @staticmethod
    def der(x):
        return np.exp2(sigmoid.run(x)) * np.exp(-x)


# relu = lambda x: np.maximum(0,x)
# relu_der = lambda x: 0 if relu(x) == 0 else 1
