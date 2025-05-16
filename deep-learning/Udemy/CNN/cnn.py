import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np

#preprocessing the train set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

training_set = train_datagen.flow_from_directory(
    'dataset/training_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

#preprocessing the test set
test_datagen = ImageDataGenerator(rescale=1./255)

test_set = test_datagen.flow_from_directory(
    'dataset/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

#initialising the CNN
cnn = tf.keras.models.Sequential()

#convolution layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3])) #input_shape=[target_size, target_size, n] --> n = 3 for colors / n = 1 for black & wihite

#pooling layer
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

#second convolution layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu')) #no need for input_shape cauz already specifyed in the first layer
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

#flattening
cnn.add(tf.keras.layers.Flatten())

#Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

#output layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

#compiling the CNN
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#training the CNN
cnn.fit(x=training_set, validation_data=test_set, epochs=25)

#reading the image want to predict
test_image = image.load_img('dataset/single_prediction/cat.jpg', target_size=(64, 64))

#converting the image to a 2D array
test_image = image.img_to_array(test_image)

#adding the batch dim
test_image = np.expand_dims(test_image, axis=0)

#making prediction and getting the result
result = cnn.predict(test_image)

# automatically assigns a numeric label to each class 1 for dog and 0 for cat
training_set.class_indices

#getting the prediction
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)