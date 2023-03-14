import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras import initializers
import keras.backend as K
import apuf_lib as ap
import challengeUtils as cts
import utils

import time
import pdb

length=64
epoch=800
batch_size= 200
nrof_chs = 800#85000

#dfx=pd.read_csv('dataset/dataset-{}bit-challenge/dataset-10m.csv'.format(length),header=None)
#dfy=pd.read_csv('dataset/dataset-{}bit-challenge/response/multimode/apuf64-ur-8mode-sft.csv'.format(length),header=None)
apuf = ap.APUF(length)

# cts.prepareDiffChs: ratio=0 means all challenges are generated randomly
rawX = utils.genNChallenges(length, nrof_chs)
#rawX = cts.prepareDiffChs(length, nrof_chs, ratio=0.85)#0.85)
X = ap.challengeTransform(rawX, length, nrof_chs)
Y = apuf.getPufResponse(X, nrof_chs, noisefree=True)

pdb.set_trace()
# fix random seed for reproducibility
#np.random.seed(42)

train_features, test_features, train_labels, test_labels = train_test_split(X, Y, test_size = 0.2)
# create model
model = Sequential()

model.add(Dense(5,input_dim=length+1, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# print the model
model.summary()

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])#,f1])

# fit the model
results = model.fit(
 train_features, train_labels,
 epochs= epoch,
 #verbose=1,
 batch_size = batch_size,
 validation_data = (test_features, test_labels)
)
# evaluvate the model
#scores = model.evaluate(test_features, test_labels)
#print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# ==================================
# Heldout test
# ==================================
nrof_heldout = 10000
h_chs = utils.genNChallenges(length, nrof_heldout)
h_pty = ap.challengeTransform(h_chs, length, nrof_heldout)
h_res = apuf.getPufResponse(h_pty, nrof_heldout, noisefree=True)

heldout_err = model.evaluate(h_pty, h_res)
print("heldout_test err = {}".format(heldout_err))
pdb.set_trace()
