from sklearn.manifold import TSNE
import numpy as np
np.random.seed(9)
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import pandas as pd
from keras.models import load_model

data = np.load("../data/data.npz")

x_test = data["x_test"]
y_test = data["y_test"]

assert x_test.shape == (234, 224, 224, 3)
assert y_test.shape == (234, 1)

x_test = x_test / 255.0

model = load_model('../models/EfficientNetB0/model00000533.h5')

res = model.predict(x_test)
y_pred = np.argmax(res, axis=1).reshape((len(x_test), 1))
y_pred = list(y_pred.reshape(-1, ))
matched = list([int(y_pred[i] == y_test[i]) for i, elem in enumerate(y_pred)])

markers = {1: "correct", 0: "incorrect"}

grad_model = tf.keras.models.Model([model.layers[0].input], [model.layers[234].output])
total_features = grad_model.predict(x_test)
features = total_features.reshape(total_features.shape[0], total_features.shape[1]*total_features.shape[2]*total_features.shape[3])

labels_adult = {0: 'unfed', 1: 'gravid', 2: "semi-gravid", 3: "fully fed"}

perplexity = np.arange(26, 27)
for per in perplexity:
    
    print(f"Claculate TSNE for per-{per}")

    tsne = TSNE(n_components=2, verbose=0, perplexity=per, n_iter=300)
    output = tsne.fit_transform(features, y_test)
    
    tsne_result_df = pd.DataFrame({'tsne_1': output[:,0], 'tsne_2': output[:,1], 'stages': [labels_adult[elem] for elem in list(y_test.reshape(-1, ))],
                                    'predicted': [markers[elem] for elem in matched]})
    
    fig, ax = plt.subplots(figsize = (8, 5))
    sns.scatterplot(x='tsne_1', y='tsne_2', hue='stages', style='predicted', data=tsne_result_df, ax=ax)
    lim = (output.min()-5, output.max()+5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    
    plt.savefig(f'tsne.png')

print("Finished")
