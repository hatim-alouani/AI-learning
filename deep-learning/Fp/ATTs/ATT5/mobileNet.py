import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split

(x_train_full, y_train_full), (x_test, y_test) = fashion_mnist.load_data()

x_train_full = x_train_full.astype('float32') / 255
x_test = x_test.astype('float32') / 255

x_train , x_val, y_train, y_val = train_test_split(x_train_full, y_train_full, test_size=0.2, random_state=42)

def create_data_generator(data, labels, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def preprocess_data(data):
    data = data.reshape(-1, 28, 28, 1)  # Reshape to 28x28x1
    data = tf.image.resize(data, (224, 224))  # Resize to 224x224
    data = tf.image.grayscale_to_rgb(data)  # Convert grayscale to RGB
    return data

x_train_pre = preprocess_data(x_train)
x_val_pre = preprocess_data(x_val)
x_test_pre = preprocess_data(x_test)

train_dataset = create_data_generator(x_train_pre, y_train)
val_dataset = create_data_generator(x_val_pre, y_val)
test_dataset = create_data_generator(x_test_pre, y_test)

#MObileNetV2 model
model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
model.trainable = False  # Freeze the base model because we are using transfer learning
model_transfer = tf.keras.Sequential()
model_transfer.add(model)
model_transfer.add(tf.keras.layers.GlobalAveragePooling2D())
model_transfer.add(tf.keras.layers.Dense(128, activation='relu'))
model_transfer.add(tf.keras.layers.Dense(10, activation='softmax'))
model_transfer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_transfer = model_transfer.fit(train_dataset, epochs=5, validation_data=val_dataset, verbose=1)
print("Training with transfer learning:")
print(history_transfer.history['accuracy'])
