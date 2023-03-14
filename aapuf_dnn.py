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
from apuf_lib import APUF, CRP


debug = False
DATA_HD_DIST = False

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
        #cts.prepareHDTChallenges(length, nrof_chs, cidx=cidx, HDT=1, ratio=ratio, PC=False)
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

def getPoisonedRes(chs, cIdx, apRes, fType='XOR', getMasked=False, xReduced = False, deviceWord = np.array([]), seqlen = 128):
    print("c_idx = {}".format(cIdx))
    c_idx = cIdx # for debug
    length = chs.shape[1]
    if len(deviceWord) == length:
        chs = np.logical_and(chs, deviceWord)
        print("Device Specific Word: {}".format(deviceWord))
    if xReduced:
        l = len(c_idx)//2
        xrf = InvertFunctions(chs)
        chs = xrf.XOR_REDUCED(c_idx, l)
        c_idx = np.arange(l)
        print("Reduced trigger size = {}".format(l))  
        #pdb.set_trace()
        
    f = InvertFunctions(chs)
    mask = None
    if fType == 'XOR':
        mask = f.XOR(c_idx)
    elif fType == 'ANDOR':
        mask = f.ANDOR(c_idx)
    elif fType == 'AND':
        mask = f.AND(c_idx)
    elif fType == 'TFF_AND':
        mask = f.TFF_AND(c_idx, seqlen).flatten()
    elif fType == 'None':
        mask = np.zeros(chs.shape[0])
    ivt_idx = np.where(mask==1)[0]
    n_ivt_idx = np.where(mask==0)[0]
    #if len(ivt_idx)==0:
        #pdb.set_trace()
    print("Inverted_idx = {}, shape={}".format(ivt_idx, ivt_idx.shape))

    pRes = apRes.copy()
    for i in ivt_idx:
       pRes[i] = not pRes[i]
    if getMasked:
        return pRes, ivt_idx, n_ivt_idx
    return pRes

# 15,50 2 layers for CCA
def trainPUFModel(in_dim, nTriggers, train_x, train_y, epoch, batch_size, test_x, test_y, nrof_layers, ftype='XOR', seqlen=128):
    model = Sequential()
    if ftype=='TFF_AND':
        train_x = train_x.reshape(-1, seqlen, in_dim)
        train_y = train_y.reshape(-1, seqlen)
        test_x = test_x.reshape(-1, seqlen, in_dim)
        test_y = test_y.reshape(-1, seqlen)
        '''
        stride = seqlen//2
        batch_size = batch_size//(stride*10)
        
        train_x = train_x.reshape(-1, in_dim)
        test_x = test_x.reshape(-1, in_dim)
        train_y = train_y.flatten()
        test_y = test_y.flatten()
        
        nrof_tr_sample = len(train_x) - seqlen
        nrof_ts_sample = len(test_x) - seqlen

        slid_train_x = np.array([train_x[i:i+seqlen] for i in np.arange(0, nrof_tr_sample, stride)])
        slid_train_y = np.array([train_y[i:i+seqlen] for i in np.arange(0, nrof_tr_sample, stride)])
        #slid_test_x = np.array([test_x[i:i+seqlen] for i in np.arange(0, nrof_ts_sample, stride)])
        #slid_test_y = np.array([test_y[i:i+seqlen] for i in np.arange(0, nrof_ts_sample, stride)])
        '''
        #pdb.set_trace()

        model.add(GRU(50,activation='relu', return_sequences=True, input_shape=(seqlen, in_dim)))
        for i in range(nrof_layers-1):
            model.add(GRU(50, activation='relu', return_sequences=True))
        #model.add(GRU(15, return_sequences=True))
        #model.add(TimeDistributed(Dense(15, activation='relu')))
        model.add(TimeDistributed(Dense(1, activation='sigmoid')))
        #model.add(Dense(1, activation='sigmoid'))
        model.summary()
        
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])#,f1])
        callback = EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=30)
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
        
    else:
        model.add(Dense(50,input_dim=in_dim, activation='relu'))
        for i in range(nrof_layers-1):
            model.add(Dense(50, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])#,f1])
        callback = EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=50)

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


# DNN attacks on APUF
np.random.seed(37487)
length = 128

nrof_chs = 2000000#2600000
invFunction = 'XOR' #ANDOR #'XOR'#'TFF_AND' #'None' for pure APUF
mode = 'HDTSampled' #SAC #HDT #B-ADD-RAND #BIASED #RANDOM
nrof_tr_bits = 8
nlayers = 2
xNetReduced = False#False#True
dSWord = np.array([]) #np.random.choice(2, length)#np.array([])

epoch = 2000
batch_size = 20000

rand_ratio = 0.7

train_test_ratio = 0.9
split_idx = int(nrof_chs*train_test_ratio)


print("========Setting====== \n-InvertFunction={}\n-Trigger bits={} \n-Number of CRPs={}\n Mode={}\n XORNet = {}\n DeviceWord={}, Rand_ratio={}, Total_Layers={}".format(invFunction, nrof_tr_bits, nrof_chs, mode, xNetReduced, dSWord, rand_ratio, nlayers))

# for T-Flip-Flop
seqlen = 128#int(math.pow(2, nrof_tr_bits))
if invFunction == 'TFF_AND':
    nrof_chs = (nrof_chs//seqlen)*seqlen
    training_batch = split_idx//seqlen
    split_idx = training_batch*seqlen
    batch_size = training_batch//20

#indices for the triggers of an AAPUF
#c_idx = np.random.choice(range(length), nrof_tr_bits, replace=False)
c_idx = np.array([2,22,31,32,72,90,100,115])
#c_idx = np.array([20,28,82,125])
c_idx.sort()

apuf = APUF(length)
#cts.prepareDiffChs: ratio=0 means all challenges are generated randomly
#chs = cts.prepareDiffChs2(length, nrof_chs, ratio=0.85)

additionalIndices = np.array([0,1,126,127])
pattern_8bit_idx = np.array([0, 1, 2, 3, 4, 7, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127])
#pattern_4bit_idx = np.array([0, 1, 2, 6, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127])
#chs = prepareChallenges(length, nrof_chs, cidx=np.append(c_idx,additionalIndices), mode = mode, ratio=rand_ratio)
#chs = prepareChallenges(length, nrof_chs, cidx=pattern_8bit_idx, mode = mode, ratio=rand_ratio)
chs = prepareChallenges(length, nrof_chs, mode = mode, ratio=rand_ratio)

'''
if DATA_HD_DIST:
    print("calculate HD dist...")
    hdDist = cts.HDDist(chs)
    print("challenge HD Distribution: {}".format(hdDist))
    print("Average HD: {}".format(cts.HDAvg(hdDist)))
'''

#pdb.set_trace()

parity = ap.challengeTransform(chs, length, nrof_chs)
res = apuf.getPufResponse(parity, nrof_chs, noisefree=True)
training_set = CRP(parity[:split_idx], res[:split_idx])
testing_set = CRP(parity[split_idx:], res[split_idx:])

# Get poisoned CRP from inversion functions
noisy_res, ivt_idx, _= getPoisonedRes(chs, c_idx, res, fType = invFunction, getMasked=True, xReduced=xNetReduced, deviceWord = dSWord, seqlen = seqlen)

training_set.response = noisy_res[:split_idx]
testing_set.response = noisy_res[split_idx:]

#pdb.set_trace()
start = time.time()
model = trainPUFModel(length+1, nrof_tr_bits, training_set.challenge, training_set.response,\
                      epoch, batch_size, testing_set.challenge, testing_set.response,\
                      nrof_layers=nlayers, ftype=invFunction, seqlen = seqlen)
t = time.time()-start
print("=====================")
print("training time: {} sec.".format(t))
print("=====================")
# ==================================
# Heldout test
# ==================================
nrof_heldout = 20000

# for T-Flip-Flop
if invFunction == 'TFF_AND':
    nrof_heldout = (nrof_heldout//seqlen)*seqlen

h_chs = utils.genNChallenges(length, nrof_heldout)
h_pty = ap.challengeTransform(h_chs, length, nrof_heldout)
h_res = apuf.getPufResponse(h_pty, nrof_heldout, noisefree=True)

h_noisy_res, h_ivt_idx, h_nIvt_idx = getPoisonedRes(h_chs, c_idx, h_res, fType = invFunction, getMasked=True, xReduced=xNetReduced, deviceWord = dSWord, seqlen = seqlen)

if invFunction == 'TFF_AND':
    h_pty = h_pty.reshape(-1, seqlen, length+1)

#pdb.set_trace()
pred = model.predict(h_pty)
y_pred = utils.softResToHard(pred).flatten()
heldout_acc = (y_pred==h_noisy_res).sum()/nrof_heldout
heldout_ivt_acc = (y_pred[h_ivt_idx]==h_noisy_res[h_ivt_idx]).sum()/(len(h_ivt_idx)+1)
heldout_non_ivt_acc = (y_pred[h_nIvt_idx]==h_noisy_res[h_nIvt_idx]).sum()/len(h_nIvt_idx)

#pdb.set_trace()

print("========Setting====== \n-InvertFunction={}\n-Trigger bits={} \n-Number of CRPs={}\n Mode={}\n XORNet = {}\n DeviceWord={}, Rand_ratio={}, Hidden_Layers={}".format(invFunction, nrof_tr_bits, nrof_chs, mode, xNetReduced, dSWord, rand_ratio, nlayers))
print("PRI in Training set (%): {}".format(len(ivt_idx)/len(noisy_res)))

print("[heldout_test] Prediction Acc = {}".format(heldout_acc))
print("[heldout_test] Acc for Inverted Responses = {}".format(heldout_ivt_acc))
print("[heldout_test] Acc for Non-Inverted Responses = {}".format(heldout_non_ivt_acc))



