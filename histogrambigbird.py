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


pred = np.load("pred_bigbird_prob.npy")
Y = np.load("testlabel.npy")
dup_prob = pred[Y[:,0]==1,1]
aut_prob = pred[Y[:,0]==0,0]
predlbl = np.argmax(pred, axis=1) ## 0 is duplicate, 1 is authentic
predlbl = 1-predlbl ## 1 is duplicate, 0 is authentic
Ylbl = np.argmax(Y, axis=1)
id = np.arange(0,len(predlbl)) ## id of all the test samples
diff = predlbl - Ylbl
id = id[diff!=0] ## id of all the test samples that are misclassified
from matplotlib import pyplot as plt
import numpy as np


# Creating histogram
fig, ax = plt.subplots(figsize=(10, 7))
ax.hist(aut_prob, bins=round(np.sqrt(len(aut_prob))))
# aut_prob1 = aut_prob[aut_prob<0.2]
# Show plot
plt.show()

data = pd.read_csv('all_random_cleaned.csv')
testid = np.load("testid.npy")
data = data.iloc[testid]

for i in id:
    print(i,":")
    print(data.iloc[i]["text_data"])
    print(pred[i])
    print(Y[i])

