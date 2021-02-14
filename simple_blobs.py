# Defines a network that can find separate data
# from two blobs of data from different classes

# Imports

from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as pyplot
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']="2"

# Helper functions

def plot_data(pl, X, y):
    #  plot class where y == 0
    pl.plot(X[y==0, 0], X[y==0, 1], 'ob', alpha=0.5)

    # plot class where y == 1
    pl.plot(X[y==1, 0], X[y==1, 1], 'x', alpha=0.5)
    pl.legend(['0', '1'])
    return pl


# def plot_decision_boundary(model, X, y):
#     amin, bmin = X.min(axis=0) - 0.1
#     amax, bmax = X.max(axis=0) + 0.1

#     hticks = np.linspace(amin, amax, 101)
#     vticks = np.linspace(bmin, bmax, 101)

#     aa, bb = np.meshgrid(hticks, vticks)
#     ab = np.<_[aa.ravel(), b.ravel()]

#     c = model.predict(ab)
#     Z = c.reshape(aa.shape)

#     plt.figure(figsize=(12, 8))
#     plt.contourf(aa, bb, Z, cmap="bw", alpha=0.2)

#     plot_data(plt, X, y)

#     return plt


X, y = make_blobs(n_samples=1000, centers=2, random_state=42)
# pl = plot_data(plt, X, y)
# pl.show()

# split the data into training and testing

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

model = Sequential()

#  a dense layer of one neurone for evaluating weather the data belongs to class 0 or 1

model.add(Dense(1, input_shape=(2,), activation="sigmoid"))
model.compile(Adam(lr=0.05), 'binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, verbose=0)
eval_result = model.evaluate(X_test, y_test)

print('\n\nTest Ions', eval_result[0], 'Test Accuracy : ', eval_result[1])