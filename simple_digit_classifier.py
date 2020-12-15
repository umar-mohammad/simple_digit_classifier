from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
import tensorflow as tf 
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

model = Sequential()
model.add(Conv2D(128, activation='relu', kernel_size=(3,3), input_size=(64,28,28)))
model.add(MaxPool2D())
model.add(Conv2D(64, activation='relu', kernel_size=(3,3)))
model.add(MaxPool2D())
model.add(Dense(10, activation='relu'))
model.compile(Adam(), metrics=['val_accuracy'])
model.fit(x_train, y_train, batch_size=64, epochs=15, verbose=2, validation_data=(x_test, y_test))
model.summary()

