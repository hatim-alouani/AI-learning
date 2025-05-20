import tensorflow as tf 
import numpy as np
from sklearn.model_selection import train_test_split

def preprocess_images(images):
    images = tf.convert_to_tensor(images[..., np.newaxis])
    images = tf.image.resize(images, [224, 224])
    images = tf.image.grayscale_to_rgb(images)
    return images

(x_train , y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

x_train , x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

x_train = preprocess_images(x_train)
x_test = preprocess_images(x_test)
x_val = preprocess_images(x_val)

def build_model(pretrained_model):
    model = tf.keras.models.Sequential()
    model.add(pretrained_model)
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(units=10, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10)

base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable= False
history = build_model(base_model)
print("Training with MobileNetV2:")
print(history.history)
