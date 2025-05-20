import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split

(x_train_full, y_train_full), (x_test, y_test) = fashion_mnist.load_data()

x_train_full = x_train_full.astype('float32') / 255
x_test = x_test.astype('float32') / 255

x_train , x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.2, random_state=42)

def create_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    return model

def train_model(optimizer):
    model = create_model()
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_val, y_val), verbose=0)

#training with adam optimizer
history_adam = train_model('adam')
#training with sgd optimizer
history_sgd = train_model('sgd')

#compare the results
print("Training with Adam optimizer:")
print(history_adam.history['accuracy'])
print("Training with SGD optimizer:")
print(history_sgd.history['accuracy'])

# #training with random weight initialization
# model_random = create_model()
# model_random.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# history_random = model_random.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_val, y_val), verbose=0)
# print("Training with random optimizer:")
# print(history_random.history['accuracy'])

