from snn.layers import softmax
from snn.utils import accuracy
from snn.utils.losses import cross_ent


class NN:
    def __init__(self, layers, loss, lr = 1E-4) -> None:
        self.layers = layers
        self.loss = loss
        self.lr  = lr

    def train(self,x,y):
        self.x = x
        self.y = y
        i, loss, accu = self.forward()
        self.backward()
        return i, loss, accu

    def forward(self):
        i = self.x
        for layer in self.layers:

            i = layer.forward(i)

        self.yhat = i
        loss = self.loss.run(y=self.y, yhat = i)
        accu = accuracy(y=self.y, yhat = i)
        return i, loss, accu

    # todo: finish backprop for entire nn
    def backward(self):
        grad = []

        if not self.loss is cross_ent:
            grad = self.loss.der(self.y, self.yhat)
        lrs = self.layers[::-1]
        for layer in lrs:
            if type(layer) is softmax:
                grad = layer.backward(self.y)
            else:
                grad = layer.update_params(grad, self.lr)
                # print(grad.shape)
            # print(grad.shape)


        




    