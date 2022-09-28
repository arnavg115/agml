from snn.nn import NN
import numpy as np
import pandas as pd
import cv2

df = pd.read_csv("train.csv")
data = df.to_numpy()
X = np.reshape(data[:,1:],(42000,784)) / 255
X_pred = np.reshape(X,(42000,784,1))

Y = np.reshape(data[:,0],(42000))
nn:NN = NN.load("model.pkl")
# inds = 20000

print(f"Validation Accuracy: {nn.validate(X[10000:],Y[10000:])}")

ind = 1040

print(f"Prediction: {np.argmax(nn.forward(X_pred[10000:][ind]))}, Actual: {Y[10000:][ind]}")
cv2.imshow("s",cv2.resize(np.reshape(X[10000:][ind],(28,28)),(500,500)))

cv2.waitKey(0)

# nn = NN.load("model.pkl")
      
# for x in X[0:1]:
#     plt.imshow()

