import numpy as np


class mse:
    name = "mse"

    @staticmethod
    def run( y, yhat):
        return np.mean(np.power(y-yhat,2))
    
    @staticmethod
    def der( y, yhat):
        return 2/np.size(y) * (yhat-y) 


class cross_ent:
    name="cross_ent"
    
    @staticmethod
    def run(y, yhat):
        return -1 * np.sum(y*np.log10(yhat))
    


