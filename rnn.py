from agml import layers, loss
import numpy as np


class SimpleRNN:
    def __init__(self, inpt_size, hidden_size, output_size):
        self.input2hid = layers.Dense(hidden_size, inpt_size + hidden_size).construct(
            kaiming=True, xavier=False, optimizer="adam"
        )
        self.input2out = layers.Dense(output_size, inpt_size + hidden_size).construct(
            kaiming=True, xavier=False, optimizer="adam"
        )

    def forward(self, inpt, hdn):
        combined = np.concatenate((inpt, hdn), axis=1)
        next_hdn = self.input2hid.forward(combined)
        out = self.input2out.forward(combined)
        return out, next_hdn

    def forward(self, otpt_grad_h, otpt_grad_y=None):
        pass


class RNN:
    def __init__(self, inpt_size, hidden_size, output_size):
        self.input = layers.Dense(hidden_size, inpt_size).construct(
            kaiming=True, xavier=False, optimizer="adam"
        )
        self.hidden = layers.Dense(hidden_size, hidden_size).construct(
            kaiming=True, xavier=False, optimizer="adam"
        )
        self.sigmoid = layers.sigmoid()
        self.tanh = layers.tanh()
        self.output = layers.Dense(output_size, hidden_size).construct(
            kaiming=True, xavier=False, optimizer="adam"
        )

    def forward(self, inpt, hdn):
        inp = self.input.forward(inpt)
        hdn_state = self.hidden.forward(hdn)
        combined = inp + hdn_state
        next_hdn = self.sigmoid.forward(combined)
        otpt = self.output.forward(next_hdn)
        return self.tanh.forward(otpt), next_hdn

    # TODO: Figure this jumbled mess out
    def backward(self, otpt_grad_h, otpt_grad_y=None):
        if otpt_grad_y is not None:
            grad = self.output.backward(otpt_grad_y, lr=0.01, avg=1)
