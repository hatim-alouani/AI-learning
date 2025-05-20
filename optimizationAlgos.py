import tensorflow as tf 
import numpy as np
from sklearn.model_selection import train_test_split

(x_train , y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

x_train , x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

def model(optimizer):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(units=10, activation='sigmoid'))
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=32)

history_adam = model('adam')
history_sgd = model('sgd')

print("Training with Adam optimizer:")
print(history_adam.history)
print("Training with SGD optimizer:")
print(history_sgd.history)