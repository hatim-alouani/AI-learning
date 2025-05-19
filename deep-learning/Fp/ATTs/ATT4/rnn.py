import numpy as np
import tensorflow as tf

x = np.linspace(0, 50, 5000) #linspace(start, stop, num=5000) is used to create an array of 5000 evenly spaced values between 0 and 50.
y = np.sin(x) + 0.1 * np.random.normal(0, 0.1, 5000) # is used to create an array of 5000 values that are the sine of x plus some noise.

# generate the data sinus and noise
import matplotlib.pyplot as plt
x = np.linspace(0, 10, 1000)
y = np.sin(x)

#sequence preparation
seq_lenght = 30
X, Y = [], []
for i in range(len(y) - seq_lenght):
    X.append(y[i:i + seq_lenght])
    Y.append(y[i + seq_lenght])

X = np.array(X).reshape(-1, seq_lenght, 1)
Y = np.array(Y).reshape(-1, 1)

#RNN model
def rnn():
    # Create a simple RNN model
    model = tf.keras.models.Sequential()
    # Add a SimpleRNN layer
    model.add(tf.keras.layers.SimpleRNN(32, activation='tanh', input_shape=(seq_lenght, 1)))
    # Add a fully connected layer
    model.add(tf.keras.layers.Dense(1))
    # Compile the model
    model.compile(optimizer='adam', loss='mse')
    return model

#LSTM model
def lstm():
    # Create a simple LSTM model
    model =tf.keras.models.Sequential()
    # Add a LSTM layer
    model.add(tf.keras.layers.LSTM(32, return_sequences=False, input_shape=(seq_lenght, 1)))
    # Add a fully connected layer
    model.add(tf.keras.layers.Dense(1))
    # Compile the model
    model.compile(optimizer='adam', loss='mse')
    return model

#GRU model
def gru():
    # Create a simple GRU model
    model = tf.keras.models.Sequential()
    # Add a GRU layer
    model.add(tf.keras.layers.GRU(32, input_shape=(seq_lenght, 1)))
    # Add a fully connected layer
    model.add(tf.keras.layers.Dense(1))
    # Compile the model
    model.compile(optimizer='adam', loss='mse')
    return model

#bidirectional lstm model
def bidirectional_lstm():
    # Create a simple Bidirectional LSTM model
    model = tf.keras.models.Sequential()
    # Add a Bidirectional LSTM layer
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16), input_shape=(seq_lenght, 1)))
    # Add a fully connected layer
    model.add(tf.keras.layers.Dense(1))
    # Compile the model
    model.compile(optimizer='adam', loss='mse')
    return model

#bidirectional GRU model
def bidirectional_gru():
    # Create a simple Bidirectional GRU model
    model = tf.keras.models.Sequential()
    # Add a Bidirectional GRU layer
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(16), input_shape=(seq_lenght, 1)))
    # Add a fully connected layer
    model.add(tf.keras.layers.Dense(1))
    # Compile the model
    model.compile(optimizer='adam', loss='mse')
    return model

models = {
    'RNN': rnn(),
    'LSTM': lstm(),
    'GRU': gru(),
    'Bidirectional LSTM': bidirectional_lstm(),
    'Bidirectional GRU': bidirectional_gru()
}

history_dict = {}
for name, model in models.items():
    # Train the model
    history = model.fit(X, Y, epochs=10, batch_size=64, verbose=0, validation_split=0.2)
    history_dict[name] = history.history

for name, history in history_dict.items():
    print(f"{name} - Loss: {history['loss'][-1]:.4f}")