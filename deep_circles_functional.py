from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
inputs = Input(shape=(2,))

from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

X, y = make_circles(n_samples=1000, factor=.6, noise=.1, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define 2 hidden layers
x = Dense(4, activation="tanh", name="Hidden-1")(inputs)
x = Dense(4, activation="tanh", name='Hidden-2')(x)

# Output layer
o = Dense(1, activation="sigmoid", name="Output_Layer")(x)

# create the model and specify the input and output
model = Model(inputs=inputs, outputs=o)

model.summary()

model.compile(Adam(lr=0.05), 'binary_crossentropy', metrics=['accuracy'])
my_callbacks = [EarlyStopping(monitor='val_accuracy', patience=5, mode=max)]
model.fit(X_train, y_train, epochs=100, verbose=1, callbacks=my_callbacks, validation_data=(X_test, y_test))
eval_result = model.evaluate(X_test, y_test)

print('\n\nTest Ions', eval_result[0], 'Test Accuracy : ', eval_result[1])
