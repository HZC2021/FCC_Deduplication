import tensorflow as tf
from keras.models import Sequential

from keras.layers import Dense, Activation  # , RNN


import os

from gcnlstm2 import GCNLSTM
# from attentionmodel import GCNLSTM
import numpy as np
from pickle import dump
from keras.regularizers import L1
import math
from sklearn.metrics import mean_squared_error, r2_score
import random
import datetime
import pandas as pd


def scheduler(epoch, lr):
    if epoch < 50:
        return lr
    elif epoch < 600:
        return lr * 1 / (1 + 5e-5 * epoch)
    elif epoch == 600:
        return 1e-3
    else:
        return lr * 1 / (1 + 5e-5 * epoch)

def create_model(units=16, drop_out=0., l1=1e-5, batch_input_shape=None):
    kernel_regularizer = L1(l1=l1)
    model = GCNLSTM(units, 16, drop_out, kernel_regularizer)
    # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2), loss="mse")
    # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2), loss="binary_crossentropy")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="categorical_crossentropy")
    return model

def trainwithval(trainset, valset, units=50, drop_out=0., batch_size=16, epochs=50, L1_value=0.):
    # design network

    model = create_model(units=units, drop_out=drop_out, l1 = L1_value)
    # batch_input_shape = [batch_size, trainset[0].shape[1], trainset[0].shape[2]])
    # define callback func

    file_name = "tmp.ckpt"
    checkpoint_path = os.path.join('./train_GCNLSTM', file_name)
    callback_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)
    callback_savemodel = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=0,
        save_weights_only=True, save_best_only=True,
        save_freq="epoch")
    # devide data for training and validation
    train_X, train_y = trainset
    val_X, val_y = valset
    # fit network

    return model

def comp(pred, gt, fn, start = 0): ##compute errors
    pred1 = np.zeros(pred.shape[0])
    for i in range(pred.shape[0]):
        if pred[i,0]>=pred[i,1]:
            pred1[i] = 1.0
    gt1 = np.zeros(gt.shape[0])
    for i in range(gt.shape[0]):
        if gt[i,0]>=gt[i,1]:
            gt1[i] = 1.0
    NGT = np.sum(gt1) ## number of duplication comments
    TN = np.sum(pred1[gt1==1.0]) ## label = pred = 1
    FN = np.sum(pred1[gt1 == 0.0]) ## label=0, pred=1
    FP = NGT - TN   ## label=1, pred=0
    TP = pred1.shape[0]- NGT -FN    ## label = pred = 0
    print(TN, TP, FN, FP)
    wrongid = []
    for i in range(pred.shape[0]):
        if pred1[i] != gt1[i]:
            wrongid.append([i+start, pred[i], gt[i]])
    np.save("%s"%fn,wrongid)

allid = np.load("testid.npy")
p1 = allid[0:round(len(allid)//2)] ## test part 1
p2 = allid[round(len(allid)//2):] ## test part 2
Xall = np.load("Xt_random.npy")  ## test X
Aall = np.load("A_random.npy")  ## test graph
Yall = np.load("Y_random.npy") ## test label
X1 = Xall[p1]
A1 = Aall[p1]
Y1 = Yall[p1]
X2 = Xall[p2]
A2 = Aall[p2]
Y2 = Yall[p2]
# X = np.load("Xt_random.npy")[500:1000]
# A = np.load("A_random.npy")[500:1000]
# Y = np.load("Y_random.npy")[500:1000]

train0 = X1[:2]
train1 = A1[:2]
trainy = Y1[:2]
train = [train0, train1]


testy = Y1
test1 = [X1, A1]
test2 = [X2, A2]


model = trainwithval([train, trainy], [train, trainy], units=32,
                     drop_out=0.0, batch_size=1, epochs=1, L1_value=0.0) ## compile model
checkpoint_path = './train_GCNLSTM/best100.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)
model.load_weights(checkpoint_path)
# testhat1 = model.predict(test1, batch_size=1)
# testhat2 = model.predict(test2, batch_size=1)
# testhat = np.concatenate((test1, test2), axis=0)
# np.save("testhat2.npy", testhat2)
testhat1 = np.load("testhat1.npy")  ## load test results of part1
testhat2 = np.load("testhat2.npy")  ## load test results of part2
testhat = np.concatenate((testhat1, testhat2), axis=0)
np.save("testhat.npy", testhat)
testy = np.concatenate((Y1, Y2), axis=0) ## get test labels for part1 + part2
np.save("testlabel.npy", testy)
fn = "wrongid_test.npy"
comp(testhat,testy,fn,0) ## compute errors
# tf.compat.v1.reset_default_graph()
# tf.keras.backend.clear_session()
