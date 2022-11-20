from scipy.special import softmax
import numpy as np
import pandas as pd

pred = np.load("pred_bigbird.npy") ## load prediction logits

m = softmax(pred, axis=1) ## get prob
predcls = np.argmax(m, axis=1) ## get predicted classifications
testset = pd.read_csv("test.csv") ## get test comment texts
a = testset['label']
c1 = np.sum(predcls[a==1]) ## label=pred = 1
e1 = np.sum(predcls[a==0]) ## label=0, predicted=1
c2 = len(predcls[a==0])-e1 ## label=pred = 0
e2 = len(predcls[a==1])-c1 ## label=1, predicted=0
print(c1, c2, e1, e2)
np.save("pred_bigbird_prob.npy",m)


pass
