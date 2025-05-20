import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.imdb.load_data(num_words=10000)

x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=100)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=100)

def simple_rnn():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=10000, output_dim=32, input_length=100))
    model.add(tf.keras.layers.SimpleRNN(32, activation='tanh'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

def bidirectional_rnn():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=10000, output_dim=32, input_length=100))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(32, activation='tanh')))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

def lstm():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=10000, output_dim=32, input_length=100))
    model.add(tf.keras.layers.LSTM(32, activation='tanh'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

def bidirectional_lstm():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=10000, output_dim=32, input_length=100))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, activation='tanh')))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

def gru():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=10000, output_dim=32, input_length=100))
    model.add(tf.keras.layers.GRU(32, activation='tanh'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)


def bidirectional_gru():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=10000, output_dim=32, input_length=100))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32, activation='tanh')))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)


history_simple_rnn = simple_rnn()
history_bidirectional_rnn = bidirectional_rnn()
history_lstm = lstm()
history_bidirectional_lstm = bidirectional_lstm()
history_gru = gru()
history_bidirectional_gru = bidirectional_gru()

print("Training with simple RNN:")
print(history_simple_rnn.history)
print("Training with bidirectional RNN:")
print(history_bidirectional_rnn.history)
print("Training with LSTM:")
print(history_lstm.history)
print("Training with bidirectional LSTM:")
print(history_bidirectional_lstm.history)
print("Training with GRU:")
print(history_gru.history)
print("Training with bidirectional GRU:")
print(history_bidirectional_gru.history)