import tensorflow as tf

import os

from gcnlstm2 import GCNLSTM
# from attentionmodel import GCNLSTM
import numpy as np
from pickle import dump
from tensorflow.keras.regularizers import L1
import math
# from sklearn.metrics import mean_squared_error, r2_score
import random
import datetime

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

    file_name = "best100.ckpt"
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

    history = model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size,  validation_data=(val_X, val_y),
                        callbacks=[callback_lr, callback_savemodel], verbose=1,
                        shuffle=False)
    return model

if __name__ == "__main__":


    train = []
    test = []

    X = np.load("Xt_random.npy")[:500] ## load feature, adjacency matrix, and label
    A = np.load("A_random.npy")[:500]
    Y = np.load("Y_random.npy")[:500]
    id = np.arange(0,500)
    FakeX = X[Y[:,0]==1]
    FakeA = A[Y[:, 0] == 1]
    FakeY = Y[Y[:, 0] == 1]
    # fakeid = id[Y[:,0]==1]
    AuthX = X[Y[:,0]==0]
    AuthA = A[Y[:, 0] == 0]
    AuthY = Y[Y[:, 0] == 0]
    # authid = id[Y[:, 0] == 0]
    # trainid = np.concatenate((fakeid[:100],authid[:100]))
    # valid = np.concatenate((fakeid[100:150], authid[100:150]))
    # allid = np.arange(0,1876)
    # trainids = np.concatenate((trainid,valid))
    # allid = np.delete(allid,trainids)
    # np.save("trainid.npy",trainid)
    # np.save("valid.npy", valid)
    # np.save("testid.npy", allid)

    TrainX = np.concatenate((FakeX[:100],AuthX[:100]), axis=0) ## get 100 dup and 100 authentic samples for training
    TrainA = np.concatenate((FakeA[:100], AuthA[:100]), axis=0)
    TrainY = np.concatenate((FakeY[:100], AuthY[:100]), axis=0)
    ValX = np.concatenate((FakeX[100:150],AuthX[100:150]), axis=0) ## get 50 dup and 50 authentic samples for validation
    ValA = np.concatenate((FakeA[100:150], AuthA[100:150]), axis=0)
    ValY = np.concatenate((FakeY[100:150], AuthY[100:150]), axis=0)
    train= [TrainX,TrainA]
    val=[ValX, ValA]

    l1 = 1e-7
    model = trainwithval([train, TrainY], [val, ValY], units=32,
                         drop_out=0.0, batch_size=1, epochs=50, L1_value= l1)
    checkpoint_path = './train_GCNLSTM/best100.ckpt'
    checkpoint_dir = os.path.dirname(checkpoint_path)
    model.load_weights(checkpoint_path)


