import os
import timeit
#from utility import train_test_split
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,Dropout, LSTM, GRU, TimeDistributed
from keras import initializers
from keras.callbacks import EarlyStopping
from keras.preprocessing.sequence import TimeseriesGenerator
import keras.backend as K

import numpy as np
import apuf_lib as ap
import challengeUtils as cts
import utils
import pdb
import random
import math
import time
from InvertFunc import InvertFunctions
from apuf_lib import XORAPUF, IPUF, CRP


debug = False

training_set = CRP(np.zeros(1), np.zeros(1))

def prepareChallenges(length, nrof_chs, cidx = np.array([]), mode = 'SAC', bias = 0.1, ratio = 0.85, shuffle=True):
    print("Prepare for Challenges... Mode = {}".format(mode))
    if mode == 'RANDOM':
        return utils.genNChallenges(length, nrof_chs)
    elif mode == 'BIASED':
        chs = utils.genNChallenges(length, nrof_chs, b = bias)
        chs = np.unique(chs, axis=0)
        chs_len = len(chs)
        while chs_len < nrof_chs:
            v = nrof_chs - chs_len
            print("Re-generate {} data for redundant challenges...".format(v))
            addi_chs = utils.genNChallenges(length, v*2, b = bias)
            chs = np.concatenate([chs, addi_chs])
            chs = np.unique(chs, axis=0)
            chs_len = len(chs)
        chs = chs[:nrof_chs]
        return chs
    elif mode == 'B-ADD-RAND':
        nrof_bch = int(nrof_chs*ratio)
        nrof_rand_c = nrof_chs - nrof_bch
        rch = utils.genNChallenges(length, nrof_rand_c)
        bch = prepareChallenges(length, nrof_bch, mode = 'BIASED', bias = bias)
        barch = np.concatenate([bch, rch])
        
        if shuffle:
            barch = cts.chShuffle(barch)
        return barch
    elif mode == 'SAC':
        #rand.15
        return cts.prepareDiffChs2(length, nrof_chs, ratio=ratio)
    elif mode == 'HDT':
        #return cts.prepareHDTChallenges(length, nrof_chs, cidx=cidx, HDT=1, ratio=ratio, PC=False)
        pc = True
        HDT_t = 4
        print("HDT-info: t={}, PC={}".format(HDT_t, pc))
        return cts.prepareHDTChallenges(length, nrof_chs, cidx=cidx, HDT=4, ratio=ratio, PC=True)
    elif mode == 'HDTSampled':
        #return cts.prepareHDTChallenges(length, nrof_chs, cidx=cidx, HDT=1, ratio=ratio, PC=False)
        HDT_t = 4
        print("HDT-Random-Sampled-info: t={}".format(HDT_t))
        return cts.prepareHDTChallenges(length, nrof_chs, cidx=cidx, HDT=HDT_t, ratio=ratio, randomSample=True)
    else:
        print("=============================================================")
        print("Exception in Function \"prepareChallenges\": No mode match.")
        pdb.set_trace()
        return cts.prepareDiffChs2(length, nrof_chs, ratio=ratio)

def getPUFInstance(length, ptype='IPUF'):
    puf = None
    pufinfo = ''
    if ptype == 'IPUF':
        xxor = 4
        yxor = 4
        xpuf = XORAPUF(length, xxor, envNSigma = 0)
        ypuf = XORAPUF(length+1, yxor, envNSigma = 0)
        puf = IPUF(xpuf, ypuf, length)
        pufinfo = '({}, {})-{}'.format(xxor, yxor, ptype)
    elif ptype == 'XORPUF':
        nxor = 4
        puf=XORAPUF(length, nxor)
        pufinfo = '{}-{}'.format(nxor, ptype)

    if puf is None:
        raise Exception("Date provided can't be in the past")
    return puf, pufinfo

def getPoisonedRes(ptype, puf, chs):
    print("Get PUF responses...")
    nchs = chs.shape[0]
    length = chs.shape[1]
    pRes = np.ones((nchs))
    intp_1_idx = np.ones((nchs))
    intp_0_idx = np.ones((nchs))

    if ptype == 'IPUF':
        trSig, pRes = puf.getPufResponse(chs, noisefree = True)
        intp_1_idx = np.where(trSig==1)[0]
        intp_0_idx = np.where(trSig==0)[0]
    elif ptype == 'XORPUF':
        pRes, apsres = puf.getPufResponse(chs, nchs, noisefree = True, isParity=False)
        trSig = np.logical_xor.reduce(apsres[:, :-1],axis=1).astype(int)
        intp_1_idx = np.where(trSig==1)[0]
        intp_0_idx = np.where(trSig==0)[0]
        #pdb.set_trace()
    #debug
    if debug:
        pdb.set_trace()
    return pRes, intp_1_idx, intp_0_idx

# 15,50 2 layers for CCA
def trainPUFModel(in_dim, train_x, train_y, epoch, batch_size, test_x, test_y, nrof_layers):
    model = Sequential()

    model.add(Dense(100,input_dim=in_dim, activation='relu'))
    for i in range(nrof_layers-1):
        model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])#,f1])
    callback = EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=100)

    results = model.fit(
     train_x, train_y,
     epochs= epoch,
     #verbose=1,
     batch_size = batch_size,
     callbacks = [callback],
     validation_data = (test_x, test_y)
    )

    scores = model.evaluate(test_x, test_y)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    return model


# DNN attacks on IPUF
np.random.seed(42)
length = 128
#'IPUF', 'XORPUF'
pufType = 'IPUF'

nrof_chs = 100000#2600000
mode = 'HDTSampled' #SAC #B-ADD-RAND #BIASED #RANDOM #HDT #HDTSampled
nlayers = 3

epoch = 1000
batch_size = 20000

rand_ratio = 1

train_test_ratio = 0.9
split_idx = int(nrof_chs*train_test_ratio)

puf, pinfo = getPUFInstance(length, ptype = pufType)

print("========Setting====== \n{} \n-Number of CRPs={}\n Mode={}\n Rand_ratio={}, Total_Layers={}".format(pinfo, nrof_chs, mode, rand_ratio, nlayers))

#cts.prepareDiffChs: ratio=0 means all challenges are generated randomly
#chs = cts.prepareDiffChs2(length, nrof_chs, ratio=0.85)

c_idx = np.array([0,1,127,126])
if pufType == 'IPUF':
    c_idx = np.array([0,1,2,64,127,126])

chs = prepareChallenges(length, nrof_chs, cidx=c_idx, mode = mode, ratio=rand_ratio)
#chs = prepareChallenges(length, nrof_chs, mode = mode, ratio=rand_ratio)

#pdb.set_trace()

# Get CRP from the PUF
# mode selection

noisy_res, ivt_idx, _= getPoisonedRes(pufType, puf, chs)

# train-test split
parity = ap.challengeTransform(chs, length, nrof_chs)
training_set = CRP(parity[:split_idx], noisy_res[:split_idx])
testing_set = CRP(parity[split_idx:], noisy_res[split_idx:])

#pdb.set_trace()
start = time.time()
model = trainPUFModel(length+1, training_set.challenge, training_set.response,\
                      epoch, batch_size, testing_set.challenge, testing_set.response,\
                      nrof_layers=nlayers)
t = time.time()-start
print("=====================")
print("training time: {} sec.".format(t))
print("=====================")
# ==================================
# Heldout test
# ==================================
nrof_heldout = 20000

h_chs = utils.genNChallenges(length, nrof_heldout)
h_pty = ap.challengeTransform(h_chs, length, nrof_heldout)

h_noisy_res, h_ivt_idx, h_nIvt_idx = getPoisonedRes(pufType, puf, h_chs)

#pdb.set_trace()
pred = model.predict(h_pty)
y_pred = utils.softResToHard(pred).flatten()
heldout_acc = (y_pred==h_noisy_res).sum()/nrof_heldout
heldout_ivt_acc = (y_pred[h_ivt_idx]==h_noisy_res[h_ivt_idx]).sum()/len(h_ivt_idx)
heldout_non_ivt_acc = (y_pred[h_nIvt_idx]==h_noisy_res[h_nIvt_idx]).sum()/len(h_nIvt_idx)

#pdb.set_trace()

print("========Setting====== \n{} \n-Number of CRPs={}\n Mode={}\n Rand_ratio={}, Total_Layers={}".format(pinfo, nrof_chs, mode, rand_ratio, nlayers))
print("Probablity of response '1' in Training set (%): {}".format(len(ivt_idx)/len(noisy_res)))

print("[heldout_test] Prediction Acc = {}".format(heldout_acc))
print("[heldout_test] Acc for Inverted Responses = {}".format(heldout_ivt_acc))
print("[heldout_test] Acc for Non-Inverted Responses = {}".format(heldout_non_ivt_acc))



