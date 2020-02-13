import random
import sys
import time

import keras
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Embedding, LSTM, Dense, Lambda, Concatenate
from keras.models import Model
import numpy as np
from keras.optimizers import Adam
from keras.preprocessing.text import one_hot
from keras.preprocessing.text import text_to_word_sequence
import os
from validate import validate
from keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.python.keras import Sequential

files = os.listdir("data")
words = ""
for f in files:
    with open("data/" + f, 'r') as file:
        data = file.read()
        words += data

words = set(text_to_word_sequence(words))
X, Y = [], []
n = 150
max_len = 0
for f in files:
    with open("data/" + f, 'r') as file:
        data = file.read()
        lines = [data[i:i + n] for i in range(0, len(data), n)]
        for line in lines:
            line = one_hot(line, round(len(words) * 1.3))

            if len(line) > max_len:
                max_len = len(line)
            X.append(line)
            Y.append(int(f.split("-")[0]))

X, Y = shuffle(X, Y)

x_left, x_right, y_out = [], [], []
matches = 0
plag = 0
for i in range(len(X)):
    if Y[i] == -1:
        continue
    if len(X[i]) < max_len:
        X[i].extend([0] * (max_len - len(X[i])))
    target = Y[i] if matches == plag or plag > matches else (0 if Y[i] == 1 else 1)
    current = random.randint(0, len(Y) - 1)
    attempts = 0
    while Y[current] != target and attempts < len(X):
        current = random.randint(0,len(Y) - 1)
        attempts += 1
    if len(X[current]) < max_len:
        X[current].extend([0] * (max_len - len(X[current])))
    x_left.append(X[i])
    x_right.append(X[current])
    if Y[i] == Y[current]:
        y_out.append(0)
        matches += 1
    else:
        y_out.append(1)
        plag += 1
    Y[i] = -1
    Y[current] = -1
print("Got " + str(matches) + " matches out of " + str(len(Y)))
print("Got " + str(plag) + " plags out of " + str(len(Y)))

x_left = np.stack(x_left)
x_right = np.stack(x_right)
y_out = np.stack(y_out)

print(x_left.shape)
print(x_right.shape)
print(y_out.shape)

x_left_train, x_left_test, x_right_train, x_right_test, y_out_train, y_out_test = train_test_split(x_left, x_right,
                                                                                                   y_out, test_size=0.1)

# np.random.seed(0)  # Set a random seed for reproducibility

# Headline input: meant to receive sequences of 100 integers, between 1 and 10000.
# Note that we can name any layer by passing it a "name" argument.
input1 = Input(shape=(max_len,))
input2 = Input(shape=(max_len,))

d1 = Dense(256, activation='sigmoid', kernel_regularizer=l2(1e-3))(input1)
d2 = Dense(256, activation='sigmoid', kernel_regularizer=l2(1e-3))(input2)
concat_layer = Concatenate()([d1, d2])
d3 = Dense(128, activation='sigmoid', kernel_regularizer=l2(1e-3))(concat_layer)
output = Dense(2, activation='sigmoid', kernel_regularizer=l2(1e-3))(d3)

model = Model(inputs=[input1, input2], outputs=output)
model.compile(optimizer=Adam(lr=.00004, clipnorm=1.), loss='sparse_categorical_crossentropy')
model.summary()

session_save_file = 'writeid-' + str(int(time.time())) + '.h5'


class SaveOnFit(keras.callbacks.Callback):
    def __init__(self):
        super(SaveOnFit, self).__init__()
        self.prev_loss = np.inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get('val_loss')
        if current < self.prev_loss and logs.get('loss') > logs.get('val_loss'):
            print("Model improved, and is not overfit.")
            print(self.prev_loss)
            print(logs.get('loss'))
            print(logs.get('val_loss'))
            self.model.save(session_save_file)


model.fit([x_left_train, x_right_train],
          y_out_train,
          batch_size=8,
          validation_split=0.1,
          epochs=100,
          callbacks=[SaveOnFit()])

c, t = validate(x_left_test, x_right_test, y_out_test, model)
print("Got " + str(c) + " out of " + str(t))
