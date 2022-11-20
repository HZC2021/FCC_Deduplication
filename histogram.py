import tensorflow as tf
from keras.models import Sequential

from keras.layers import Dense, Activation  # , RNN


import os

# from mgcnlstm2 import GCNLSTM
import numpy as np
from pickle import dump
from keras.regularizers import L1
import math
from sklearn.metrics import mean_squared_error, r2_score
import random
import datetime
import pandas as pd



testhat = np.load("testhat.npy") ##load prediction result from GCN
Y = np.load("testlabel.npy") ## load ground truth
dup_prob = testhat[Y[:,0]==1,0] ## get the predicted probability of duplicates
aut_prob = testhat[Y[:,0]==0,1] ## get the predicted probability of authentic

from matplotlib import pyplot as plt
import numpy as np


# Creating histogram
fig, ax = plt.subplots(figsize=(10, 7))
ax.hist(aut_prob, bins=round(np.sqrt(len(aut_prob)))) ## plot the histogram of authentic comments
# aut_prob1 = aut_prob[aut_prob<0.2]
# Show plot
plt.show()

data = pd.read_csv('all_random_cleaned.csv') ## load the dataset
testid = np.load("testid.npy")
data = data.iloc[testid]
fn = "wrongid_test.npy" ## load the wrong prediction id
wrongset=np.load(fn, allow_pickle=True)

for i in wrongset:
    # print(data.iloc[i[0], 1])
    # print(i[1])
    # print(i[2])
    # print(i[0])
    if i[2][0]==0.0:
        if i[1][1]<0.5:
            print(data.iloc[i[0], 1])
            print(i[1])
            print(i[2])
            print(i[0])