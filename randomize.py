import numpy as np

fake_a_pad = np.load("fake_a_pad.npy").astype(np.float16) ## load the fake data and compress
fake_x_pad = np.load("fake_x_pad.npy").astype(np.float16)
fakey = np.zeros((len(fake_x_pad),2))
fakey[:,0] = 1.0 # generate binary labels
auth_a_pad = np.load("auth_a_pad.npy").astype(np.float16)
auth_x_pad = np.load("auth_x_pad.npy").astype(np.float16)
authy = np.zeros((len(auth_x_pad),2))
authy[:,1] = 1.0

At = np.concatenate((fake_a_pad, auth_a_pad), axis=0) ## concatenate the fake and authentic data
Xt = np.concatenate((fake_x_pad, auth_x_pad), axis=0)
Y = np.concatenate((fakey, authy), axis=0)
# np.save(r"Y.npy",Y)
# np.save(r"A.npy", At)
# np.save(r"Xt.npy", Xt)

# v_min = Xt.min(axis=(0, 1, 2), keepdims=True)
# v_max = Xt.max(axis=(0, 1, 2), keepdims=True)
# X_std = (Xt - v_min)/(v_max - v_min)


# np.save(r"X_norm.npy",X_std.astype(np.float16))


# randomize
# randomid = np.load(r"D:\data\randomid.npy")
# Total = len(At)
# randomid = np.random.choice(Total, Total, replace=False)
randomid = np.load("randomid.npy") ## load the random id and randomize the dataset.
# np.save("randomid.npy", randomid)
# np.save("A_random.npy",At[randomid])
# np.save("X_random.npy",X_std[randomid].astype(np.float16))
# np.save("Xt_random.npy",Xt[randomid].astype(np.float16))
# np.save("Y_random.npy",Y[randomid])

import pandas as pd
data = pd.read_csv('all.csv') ## load all comment text
newdata = data.iloc[randomid] ## randomize the comments set
lbl = Y[randomid]
blbl = np.argmin(lbl, axis=1) ## add the binary labels for bigbird
newdata['label'] = blbl.tolist()
newdata.to_csv('all_random.csv',index=False)