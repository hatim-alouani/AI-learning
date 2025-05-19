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


# Preprocess function (simple)
def preprocess_image(image, label):
    image = tf.expand_dims(image, -1)  # (28, 28, 1)
    image = tf.image.resize(image, [224, 224])
    image = tf.image.grayscale_to_rgb(image)
    return image, label

# Dataset function
def create_dataset(images, labels, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

train_dataset = create_dataset(x_train_pre, y_train)
val_dataset = create_dataset(x_val_pre, y_val)
test_dataset = create_dataset(x_test_pre, y_test)

#MObileNetV2 model
model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
model.trainable = False  # Freeze the base model because we are using transfer learning
model_transfer = tf.keras.Sequential()
model_transfer.add(model)
model_transfer.add(tf.keras.layers.GlobalAveragePooling2D())
model_transfer.add(tf.keras.layers.Dense(128, activation='relu'))
model_transfer.add(tf.keras.layers.Dense(10, activation='softmax'))
model_transfer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_transfer = model_transfer.fit(train_dataset, epochs=1, validation_data=val_dataset, verbose=1)

print("Training with transfer learning:")
print(history_transfer.history['accuracy'])


learning_rates = [0.001, 0.01, 0.1]
for lr in learning_rates:
    model_transfer.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history_transfer = model_transfer.fit(train_dataset, epochs=1, validation_data=val_dataset, verbose=1)
    print(f"Learning rate: {lr}")
    print(history_transfer.history['accuracy'])

batch_sizes = [16, 32, 64]
for batch_size in batch_sizes:
    model_bs = create_model()
    model_bs.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history_bs = model_bs.fit(train_dataset.batch(batch_size), epochs=5, validation_data=(x_val, y_val), verbose=1)
    print(f"Batch size: {batch_size}")
    print(history_bs.history['accuracy'])

