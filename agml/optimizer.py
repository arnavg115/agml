import numpy as np

class basic:
    def __init__(self) -> None:
        pass
    
    def step(self, theta_curr):
        return theta_curr


class momentum:
    def __init__(self,beta=0.9):
        self.beta = beta
        self.v_curr = 0

    def step(self, theta_curr):
        self.v_curr = self.beta * self.v_curr + (1-self.beta)*theta_curr
        return self.v_curr

class RMSprop:
    def __init__(self, beta=0.95, epsilon=1e-8) -> None:
        self.beta = beta
        self.s_curr = 0
        self.epsilon = 1e-8
    
    def step(self, theta_curr):
        self.s_curr = self.beta * self.s_curr +(1-self.beta)*(np.power(theta_curr, 2))
        return theta_curr / np.sqrt(self.s_curr + self.epsilon)
    
class adam:
    def __init__(self, beta1=0.9, beta2=0.9, epsilon=1e-8) -> None:
        self.beta1 = beta1
        self.beta2 = beta2
        self.v_curr = 0
        self.s_curr = 0
        self.epsilon = epsilon

    def step(self, theta_curr):
        self.s_curr = self.beta2 * self.s_curr +(1-self.beta2)*(np.power(theta_curr, 2))
        self.v_curr = self.beta1 * self.v_curr + (1-self.beta1)*theta_curr
        
        return self.v_curr / np.sqrt(self.s_curr + self.epsilon)
        
        

