import numpy as np 
import tensorflow as tf

def convolution_output_size(input_size, kernel_size, stride, padding):
    return (input_size + 2 * padding - kernel_size) // stride + 1

input_size = 32
kernel_size = 3
stride = 1
padding = 0
output_size = convolution_output_size(input_size, kernel_size, stride, padding)
print(f"Output size: {output_size}×{output_size}")


input_image = tf.constant([
    [1, 2, 3, 0, 1],
    [0, 1, 2, 3, 4],
    [4, 5, 6, 1, 0],
    [0, 1, 0, 2, 3],
    [3, 4, 1, 2, 1],
], dtype=tf.float32)

input_batch = tf.reshape(input_image, (1, 5, 5, 1))  #  (batch, height, width, channels)

kernel = tf.constant([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1],
], dtype=tf.float32)

kernel = tf.reshape(kernel, (3, 3, 1, 1))  # (height, width, in_channels, out_channels)

conv_valid = tf.nn.conv2d(input_batch, kernel, strides=[1, 1, 1, 1], padding='VALID')
conv_same = tf.nn.conv2d(input_batch, kernel, strides=[1, 1, 1, 1], padding='SAME')

print("Convolution with VALID padding:\n", conv_valid.numpy().squeeze())
print("Convolution with SAME padding:\n", conv_same.numpy().squeeze())


def relu(x):
    return np.maximum(0, x)
def sigmoid(x):
    return 1/(1 + np.exp(-x))
def tanh(x):
    return np.tanh(x)

x = np.array([-2., -1., 0., 1., 2.])
print("ReLU:", relu(x))
print("Sigmoid:", sigmoid(x))
print("Tanh:", tanh(x))


input_pool = tf.constant([
[1, 3, 2, 1],
[4, 6, 5, 3],
[7, 8, 9, 4],
[2, 3, 1, 0]
], dtype=tf.float32)

input_pool = tf.reshape(input_pool, (1, 4, 4, 1))  # (batch, height, width, channels)

output_max = tf.nn.max_pool2d(input_pool, ksize=2, strides=2, padding='VALID')
print("Max pooling:\n", output_max.numpy().squeeze())

output_avg = tf.nn.avg_pool2d(input_pool, ksize=2, strides=2, padding='VALID')
print("Average pooling:\n", output_avg.numpy().squeeze())

donnees = np.random.randn(10, 3) * 2 + 5
print("Donnees brutes:\n", donnees)

batch_norm = tf.keras.layers.BatchNormalization()
donnees_norm = batch_norm(donnees)

print("Avant normalisation - Moyenne:\n", np.mean(donnees, axis=0))
print("Après normalisation - Moyenne:\n", np.mean(donnees_norm, axis=0))

