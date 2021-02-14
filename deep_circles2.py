from sklearn.datasets import make_circles
X, y = make_circles(n_samples=1000, factor=.6, noise=.1, random_state=42)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

model = Sequential()

#  a dense layer of one neurone for evaluating weather the data belongs to class 0 or 1
model.add(Dense(4, input_shape=(2,), activation="tanh", name="Hidden-1"))
model.add(Dense(4, activation="tanh", name='Hidden-2'))

model.add(Dense(1, activation="sigmoid", name="Output_Layer"))
model.summary()
model.compile(Adam(lr=0.05), 'binary_crossentropy', metrics=['accuracy'])
my_callbacks = [EarlyStopping(monitor='val_accuracy', patience=5, mode=max)]
model.fit(X_train, y_train, epochs=100, verbose=1, callbacks=my_callbacks, validation_data=(X_test, y_test))
eval_result = model.evaluate(X_test, y_test)

print('\n\nTest Ions', eval_result[0], 'Test Accuracy : ', eval_result[1])

# MODEL FEATURES

# summary() - provides all the details of the model, the laters, shape of trainable parameters
# plot_model model layer hirachy

# from keras.utils import plot_model

# plot_model(model, show_shapes=True, show_layer_names=True)

# define Early stopping callbacks
