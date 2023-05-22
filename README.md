# agml

### What is this?

This is a simple neural network framework built on top of `numpy`. It is an amalgamation of many different tutorials.

### What has been implemented?

- Layers: Dense, Dropout, Softmax, Embedding
- Activations: Relu, Sigmoid, Leaky Relu, Tanh
- Loss Functions: Mean Squared Error, Softmax
- Optimizers: Momentum, Adam, RMSProp
- Features: Saving/Loading model weights, simple grayscale ascii visualizer

### What is planned?
- Layers: Convolutional
- Networks: RNN, LSTM, transformers?

### Why?

This is mostly for educational purposes as it is not as performant as the bigger frameworks

### Demo

Check out this [Demo](https://replit.com/@garnavaurha/snn?v=1) of MNIST digit recognition.

### Uses

```python
import numpy as np
from snn.layer import Dense, Relu
from snn.nn import NN
from snn.loss import mse
from snn.utils import one_hot_func

# Grab data
# X is of shape (10000,784), only 2 dims supported
# Y is of shape (10000,). In this example it has 10 labels
X = samples[0:10000]
Y = labels[0:10000]

# One hot encoding the labels. Output shape is (10000,10)
one_hot = one_hot_func(Y)

# Initialize network. Dense layers have (neurons, inpt_features)
# kaiming=True means the network uses kaiming initialization. Setting xavier=True uses xavier initialization instead
# Specifying optimizer is also allowed. 3 optimizers avalaible: momentum, adam, rmsprop.
network = NN([Dense(20,784), Relu(), Dense(10,20)], loss = mse(), kaiming=True, optimizer="rmsprop")

# Batch_size can be set.
network.train(X, one_hot, batch_size=200)

# Run inference on an image
predicted = np.argmax(network.predict(sample[10001]))

network.save("model.pkl")
```
