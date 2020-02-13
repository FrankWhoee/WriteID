import time

import keras
import numpy as np
from keras.layers import Concatenate
from keras.models import load_model
from keras_preprocessing.text import text_to_word_sequence, one_hot
from sklearn.model_selection import train_test_split
import os
import re

from sklearn.utils import shuffle



def validate(x_left, x_right, y, model, verbose=0):
    from sklearn.metrics import roc_curve
    from sklearn.metrics import auc
    actual = []
    pred = []
    total = 0
    correct = 0
    for xl, xr, yout in zip(x_left, x_right, y):
        xl = xl.reshape((1, xl.shape[0]))
        xr = xr.reshape((1, xr.shape[0]))
        prediction = model.predict(x=[xr, xl])
        if verbose == 1:
            print("-------------------------")
            print("Prediction: " + str(prediction))
            print("Actual: " + str(yout))
        total += 1
        if (prediction[0][1] > prediction[0][0] and yout == 1) or (
                prediction[0][1] < prediction[0][0] and yout == 0):
            correct += 1
        actual.append(yout)
        pred.append(1 if prediction[0][1] > prediction[0][0] else 0)
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(actual,pred)
    auc_keras = auc(fpr_keras, tpr_keras)
    print("AUC: " + str(auc_keras))
    print("FPR: " + str(fpr_keras))
    print("TPR: " + str(tpr_keras))
    return correct, total

def cross_validate(x_left, x_right, y_out, A_test, X_test, E_test, y_test):
    from keras.layers import Input, Dense
    from keras.models import Model
    import numpy as np
    from keras.optimizers import Adam
    from validate import validate
    seed = 7
    np.random.seed(seed)
    cvscores = []
    k_folds = 6
    i = 0
    data_size = x_left.shape[0]
    print("{} points of data.".format(data_size))
    while i <= int(data_size/k_folds) * (k_folds - 1):
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

        model.fit([x_left_train, x_right_train],
                  y_out_train,
                  batch_size=8,
                  validation_split=0.1,
                  epochs=50)
        # evaluate the model
        correct, total = validate( A_test,X_test, E_test, y_test, model)
        print("%s: %.2f%%" % ("accuracy", (correct/total) * 100))
        cvscores.append((correct/total) * 100)
        i += int(data_size/k_folds)
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

# files = os.listdir("data")
# words = ""
# for f in files:
#     with open("data/" + f, 'r') as file:
#         data = file.read()
#         words += data
#
# words = set(text_to_word_sequence(words))
# X, Y = [], []
# n = 150
# max_len = 0
# for f in files:
#     with open("data/" + f, 'r') as file:
#         data = file.read()
#         lines = [data[i:i + n] for i in range(0, len(data), n)]
#         for line in lines:
#             words = one_hot(line, round(len(words) * 1.3))
#             if len(words) > max_len:
#                 max_len = len(words)
#             X.append(words)
#             Y.append(int(f.split("-")[0]))
#
# X, Y = shuffle(X, Y)
#
# x_left, x_right, y_out = [], [], []
# matches = 0
# total = 0
# for i in range(0, len(X), 2):
#     if len(X[i]) < max_len:
#         X[i].extend([0] * (max_len - len(X[i])))
#     try:
#         if len(X[i + 1]) < max_len:
#             X[i + 1].extend([0] * (max_len - len(X[i + 1])))
#         x_right.append(X[i + 1])
#         x_left.append(X[i])
#     except:
#         break
#     if Y[i] == Y[i + 1]:
#         y_out.append(0)
#         matches += 1
#     else:
#         y_out.append(1)
#     total += 1
# print("Got " + str(matches) + " matches out of " + str(total))
#
# x_left = np.stack(x_left)
# x_right = np.stack(x_right)
# y_out = np.stack(y_out)
#
# print(x_left.shape)
# print(x_right.shape)
# print(y_out.shape)
#
# x_left_train, x_left_test, x_right_train, x_right_test, y_out_train, y_out_test = train_test_split(x_left,x_right,y_out, test_size=0.1)
#
# c,t = validate(x_left_test,x_right_test,y_out_test, load_model('WriteID-0.h5'))
# print("Got " + str(c) + " out of " + str(t))