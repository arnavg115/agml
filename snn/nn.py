from snn.layers import softmax
from snn.utils.losses import cross_ent


class NN:
    def __init__(self, layers, loss) -> None:
        self.layers = layers
        self.loss = loss

    def train(self,x,y):
        self.x = x
        self.y = y

    def forward(self):
        i = self.x
        for layer in self.layers:

            i = layer.forward(i)

        self.yhat = i
        loss = self.loss.run(y=self.y, yhat = i)
        return i, loss

    # todo: finish backprop for entire nn
    def backward(self):
        grad = []
        ignore = True
        if not (type(self.loss) is cross_ent):
            grad = self.loss.der(self.y, self.yhat)

        elif type(self.layers[-1]) is softmax:
            grad = self.layers[-1].backward(self.y)

        




    