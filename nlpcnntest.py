import keras
from keras.datasets import imdb
from keras.preprocessing import sequence

import tensorflow as tf
from keras.models import Sequential, Model
from keras import layers
from keras.optimizers import RMSprop

max_features = 10000  # number of words to consider as features
max_len = 500  # cut texts after this number of words (among top max_features most common words)

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

input_tensor = keras.Input(shape=(500,), name='text')
x = layers.Embedding(max_features, 128, input_length=max_len)(input_tensor)
x = layers.Convolution1D(32, 7, activation='relu')(x)
x = layers.MaxPool1D(5)(x)
x = layers.Convolution1D(32, 7, activation='relu')(x)
x = layers.GlobalMaxPool1D()(x)
output_tensor = layers.Dense(1, activation='relu')(x)
model = Model(input_tensor, output_tensor)

model.summary()

model.compile(optimizer=RMSprop(lr=1e-4),
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2)
