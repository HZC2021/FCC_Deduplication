

import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras import backend as K
from keras import activations, initializers, constraints, regularizers
from keras.layers import Input, Layer, Dropout, LSTM, Dense, Permute, Reshape, Flatten, Bidirectional, Concatenate
from keras import Model
from spektral.layers import GCNConv

class GCNLSTM(Model):
    def __init__(self, nhid, nclass, dropout = 0.3, kernel_regularizer=None):
        super(GCNLSTM, self).__init__()

        self.dropout = Dropout(dropout)

        # self.gcn2 = GCNConv(channels=nclass)
        self.lstm = LSTM(units=nclass, dropout=dropout, activation='tanh', kernel_regularizer=kernel_regularizer)
        self.nhid = nhid
        self.den = tf.keras.layers.Dense(2,activation='softmax')
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.gcn1 = GCNConv(channels=nhid, activation='relu', kernel_regularizer=kernel_regularizer)

    def call(self, inputs, training=None):
        feats, adj = inputs
        # print(feats.shape)
        # print(adj.shape)
        out = []
        # print(adj.shape[-1])
        nd = adj.shape[-1]
        #lstm loops through the inputs
        for i in range(adj.shape[1]):
            x_1 = self.gcn1([feats[:,i,:,:], adj[:,i,:,:]])
            x_1 = self.dropout(x_1)
            # dropout = self.dropout(x_1, training=training)
            h = Reshape((-1, nd * self.nhid))(x_1)
            out.append(h)
            # print(h.shape)
        H = Concatenate(axis=1)([out[i] for i in range(len(out))])
        # out = tf.stack(out, 1)
        # print(H.shape)
        # h = Reshape((-1, 67, 640))(H)
        # out2 = Permute((1, 3, 2))(h)
        # print(out2.shape)
        # H = tf.keras.activations.softmax(H)
        # print(H.shape)

        outlstm = self.lstm(H)

        # print(outlstm.shape)
        # pred = tf.keras.activations.softmax(self.den(h))

        pred = self.den(outlstm)
        # pred = tf.keras.activations.softmax(self.den(outlstm))
        return pred