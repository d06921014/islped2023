import numpy as np
import pandas as pd
import apuf_lib as ap
import pdb
import random
import utils
from apuf_lib import APUF, XORAPUF, CRP, IPUF
import matplotlib.pyplot as plt
from InvertFunc import InvertFunctions

debug = False
save = False

def getPUFInstance(length, pType='AAPUF'):
    puf = None
    pufinfo = ''
    if pType == 'IPUF':
        xxor = 3
        yxor = 3
        xpuf = XORAPUF(length, xxor, envNSigma = 0)
        ypuf = XORAPUF(length+1, yxor, envNSigma = 0)
        puf = IPUF(xpuf, ypuf, length)
        pufinfo = '({}, {})-{}'.format(xxor, yxor, pType)
    elif pType == 'XORPUF':
        nxor = 5
        puf = XORAPUF(length, nxor)
        pufinfo = '{}-{}'.format(nxor, pType)
    elif pType == 'AAPUF':
        puf = APUF(length)
        pufinfo = '{}-'.format(pType)
    elif pType == 'MMPPUF':
        puf = APUF(length)
        pufinfo = '{}-'.format(pType)

    if puf is None:
        raise Exception("No PUF tpye matched!")
    return puf, pufinfo

def getPUFResponse(puf, chs, pType='XORPUF'):
    response = None
    pufinfo = ''
    if pType == 'IPUF':
        _, response = puf.getPufResponse(chs, noisefree = True)
    elif pType == 'XORPUF':
        nrof_chs = chs.shape[0]
        response, _ = puf.getPufResponse(chs, nrof_chs, noisefree = True, isParity=False)
    elif pType == 'AAPUF':
        tiggerType= 'XOR'
        #cidx = np.array([2,22,31,32,72,90,100,115])
        cidx = np.array([20,28,82,125])
        response = getAAPUFResponse(puf, chs, cidx, tsType=tiggerType)
        pufinfo = '{}-{}'.format(tiggerType, cidx)
    elif pType == 'MMPPUF':
        tiggerType= 'XOR'
        cidx = np.array([2,22,31,32,72,90,100,115])
        response = getMMPPUFResponse(puf, chs, cidx, tsType='XOR')
        pufinfo = '{}-{}'.format(tiggerType, cidx)

    return response, pufinfo
    
def getMMPPUFResponse(apuf, chs, c_idx, tsType='XOR'):
    nrof_chs = chs.shape[0]
    length = chs.shape[1]

    shiftidx = c_idx
    sftNbits = 8
    parity = np.zeros((nrof_chs, length+1))

    for i, c in enumerate(chs):
        triggers = np.logical_xor.reduce(c[shiftidx])
        if triggers :
            c = np.roll(c, sftNbits)
        else :
            if leftsh:
                c = np.roll(c, -sftNbits)
        parity[i] = ap.challengeTransform(c.reshape(-1,length), length, nrows = 1)
    if debug:
            print("% of shift right mode: {}".format(np.logical_xor.reduce(chs[:, shiftidx], axis=1)))
    res = puf.getPufResponse(parity, nrof_chs)
    return poised_res


def getAAPUFResponse(apuf, chs, c_idx, tsType='XOR'):
    nrof_chs = chs.shape[0]
    length = chs.shape[1]
    parity = ap.challengeTransform(chs, length, nrof_chs)
    res = apuf.getPufResponse(parity, nrof_chs, noisefree=True)
    f = InvertFunctions(chs)
    mask = np.zeros(nrof_chs)
    if tsType == 'XOR':
        mask = f.XOR(c_idx)
    elif tsType == 'ANDOR':
        mask = f.ANDOR(c_idx.reshape(-1,2))
    elif tsType == 'TFF_AND':
        mask = f.TFF_AND(c_idx).flatten()
    elif tsType == 'AND':
        mask = f.AND(c_idx)
    elif tsType == 'NONE':
        pass
    else:
        print('getAAPUFResponse: No Trigger Type matched. set Trigger Type to \'XOR\'')
        mask = f.XOR(c_idx)
    
    ivt_idx = np.where(mask==1)[0]
    #print("Inversion Function:{}\nc_idx = {}\nInverted_idx = {}, shape={}".format(tsType, c_idx, ivt_idx, ivt_idx.shape))
    poised_res = getPoisonedRes(res, ivt_idx)
    return poised_res

def getPoisonedRes(r, invert_idx):
    p_res = r.copy()
    for i in invert_idx:
       p_res[i] = not p_res[i]
    return p_res

def showPlt(length, trans, nchal):
    plt.ylim(0, 1)
    plt.plot(np.arange(length), trans/nchal)
    plt.show()

if __name__ == "__main__":

    length = 128
    nchal = 1280
    #For AAPUF
    nrof_tr_bits = 8

    pufType ='AAPUF'

    chs = utils.genNChallenges(length, nchal)
    chflip = np.tile(chs, (length, 1, 1))

    # for each index in filp array, for each challenge, flip the index-th bit
    for lidx, cs in enumerate(chflip):
        for nidx, c in enumerate(cs):
            chflip[lidx, nidx, lidx] = not c[lidx]
    #pdb.set_trace()
    sacfpRes = np.zeros((length, nchal))
    trans = np.zeros((length))
    #======================
    # init PUF Instances   
    np.random.seed(11)
    puf, pufInfo1 = getPUFInstance(length, pType=pufType)
    #apuf = XORAPUF(length, 4)
    #ts_idx = np.array(random.sample(range(length), nrof_tr_bits))
    fpres, pufInfo2 = getPUFResponse(puf, chs, pType=pufType)
    #tiggers_idx = np.array([2,22,31,32,72,90,100,115])
    #fpres = getAAPUFResponse(apuf, chs, tiggers_idx, tsType=trigger_type)
    print("PUF-Info: {}".format(pufInfo1+pufInfo2))
    #=====================
    for i in range(length):
        print("{}-th index...".format(i))
        #ap.dfpSelectedByChallengeSft(apuf, chflip[i], nchal, length)
        #sacfpRes = getAAPUFResponse(apuf, chflip[i], ts_idx, tsType=trigger_type)
        #sacfpRes[i] = ap.dualFlip(apuf, chflip[i], nchal, length, np.arange(56,64))
        sacfpRes, _ = getPUFResponse(puf, chflip[i], pType=pufType)
        #pdb.set_trace()
        trans[i] = (sacfpRes != fpres).sum()
    showPlt(length, trans, nchal)
    #pdb.set_trace()
    np.savetxt("output_transition/{}-transProb.csv".format(pufInfo1+pufInfo2), trans.reshape(length,1)/nchal, fmt="%.4f", delimiter=',')
