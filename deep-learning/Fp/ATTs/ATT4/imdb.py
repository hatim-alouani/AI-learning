import tensorflow as tf

max_features = 10000

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=max_features)
#imdb is a dataset of 50,000 movie reviews from IMDB, labeled by sentiment (positive/negative). The reviews have been preprocessed and encoded as sequences of integers. The num_words parameter limits the vocabulary to the top 10,000 most frequent words.

x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=100)
# The pad_sequences function is used to ensure that all sequences in the dataset have the same length. In this case, all sequences are padded to a maximum length of 100. This is important for training models that require fixed-length input sequences.
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=100)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(max_features, 128))
# The Embedding layer is used to convert the integer-encoded words into dense vectors of fixed size (128 in this case). This is a common practice in NLP tasks to represent words in a continuous vector space.
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)))
# The Bidirectional layer is used to create a bidirectional LSTM model. This means that the LSTM processes the input sequences in both forward and backward directions, which can help capture context from both sides of the sequence.
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))
# Another Bidirectional LSTM layer is added, but this time it does not return sequences. This is typically done to reduce the dimensionality of the output before passing it to the next layer.
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
# The Dense layer is a fully connected layer that outputs a single value (0 or 1) with a sigmoid activation function. This is suitable for binary classification tasks like sentiment analysis.

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)
model.summary()
#summary() is used to print a summary of the model architecture, including the number of parameters in each layer and the total number of parameters in the model.
