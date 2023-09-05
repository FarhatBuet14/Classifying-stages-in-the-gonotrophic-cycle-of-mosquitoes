from keras.models import load_model
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

data = np.load("../data/data.npz")

x_test = data["x_test"]
y_test = data["y_test"]
assert x_test.shape == (234, 224, 224, 3)
assert y_test.shape == (234, 1)
x_test = x_test / 255.0

model = load_model('model00000533.h5')

res = model.predict(x_test)
res_test = np.argmax(res, axis=1).reshape((234, 1))

cm_ = confusion_matrix(y_test,res_test)
accuracy_ = (sum(res_test == y_test) / len(y_test))[0] * 100
print(classification_report(y_test, res_test))
