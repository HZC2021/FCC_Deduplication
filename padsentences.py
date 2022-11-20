import os.path

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
# from graph4nlp.pytorch.modules.graph_construction.dependency_graph_construction import DependencyBasedGraphConstruction
from stanfordcorenlp import StanfordCoreNLP
import numpy as np
import spacy
import re

data = pd.read_csv('fake_bk.csv') ## duplication comment texts
sent = 67
fake_a_pad = np.zeros((1365, sent, 67, 67)) ## samples, sentences, words, words
fake_x_pad = np.zeros((1365, sent, 67, 768)) ## samples, sentences, words, vector
cnt = 0
fakeid = []

for i in range(1703):
    if os.path.exists("fake/fake_adj_arc_in_%d.npy"%i): ## check path for each comment
        fakeid.append(i)
        fake_a = np.load("fake/fake_adj_arc_in_%d.npy" %i) ## load adjacency matrix for each comment
        fake_x = np.load("fake/fake_vec_in_%d.npy" %i) ## load feature for each comment
        start_ind = sent - fake_x.shape[0] ## get the start index
        fake_a_pad[cnt, start_ind:] = fake_a ## pad and concatenate adjacency matrix
        fake_x_pad[cnt, start_ind:] = fake_x ## pad and concatenate feature
        cnt += 1
# np.save("fake_a_pad.npy", fake_a_pad)
# np.save("fake_x_pad.npy", fake_x_pad)
fakedata = data.iloc[fakeid]
np.save("fakeid.npy", fakeid)

cnt = 0
data = pd.read_csv('wait_bk.csv')
authid = []
auth_a_pad = np.zeros((512, sent, 67, 67))
auth_x_pad = np.zeros((512, sent, 67, 768))
for i in range(1297):
    if os.path.exists("auth/auth_adj_arc_in_%d.npy"%i): ## check path
        authid.append(i)
        auth_a = np.load("auth/auth_adj_arc_in_%d.npy" %i)
        auth_x = np.load("auth/auth_vec_in_%d.npy" %i)
        start_ind = sent - auth_x.shape[0]
        auth_a_pad[cnt, start_ind:] = auth_a
        auth_x_pad[cnt, start_ind:] = auth_x
        cnt += 1
# np.save("auth_a_pad.npy", auth_a_pad)
# np.save("auth_x_pad.npy", auth_x_pad)
authdata = data.iloc[authid]
np.save("authid.npy", authid)

all = pd.concat([fakedata, authdata])
all.to_csv("all.csv", index=False)