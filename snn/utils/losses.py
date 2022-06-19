import numpy as np


class mse:
    name = "mse"

    @staticmethod
    def run( y, yhat):
        return np.sum(np.exp2(y-yhat)) * (1/y.shape[0])
    
    @staticmethod
    def der( y, yhat):
        return (-2/y.shape[0]) * (y-yhat)


class cross_ent:
    name="cross_ent"
    
    @staticmethod
    def run(y, yhat):
        return -1 * np.sum(y*np.log10(yhat))
    


