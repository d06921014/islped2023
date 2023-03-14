import numpy as np
import apuf_lib as ap
import utils
import pdb
import math
import random
from apuf_lib import APUF, CRP
from itertools import combinations
import operator as op
from functools import reduce

debug = False

def comb(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom  # or / in Python 2

# bchal = np.random.choice(2, length*nrof_chs, p=[0.1, 0.9]).reshape(-1, length)
def prepareDiffChs(length, nrof_chs, ratio=0.5):
    nrof_bch = int(ratio*nrof_chs)
    #bchal = np.zeros(length).reshape(-1, length)
    bchal = np.random.choice(2, length).reshape(-1, length)
    while len(bchal) < nrof_bch:
        bchal = utils.genChallengesBySAC(length, len(bchal), chs=bchal)
        bchal = np.unique(bchal, axis=0)
        if (len(bchal)*length > 3000000) and (len(bchal) < nrof_bch):
            bchal = bchal[:50000]
            bchal = utils.genChallengesBySAC(length, len(bchal), chs=bchal)
            bchal = np.unique(bchal, axis=0)
            break
        print("bchal.shape = {}".format(bchal.shape))
    bchal = bchal[:nrof_bch]
    if debug:
        pdb.set_trace()
    if ratio != 1:
        nrof_rand_c = nrof_chs - nrof_bch
        rchal = utils.genNChallenges(length, nrof_rand_c)
        bchal = np.concatenate([bchal, rchal])
    #Shuffle
    chidx = np.arange(nrof_chs)
    np.random.shuffle(chidx)
    bchal = bchal[chidx]
    return bchal

def prepareDiffChs2(length, nrof_chs, init_n_chs=1, ratio=0.5, shuffle = True):
    #np.random.seed(42)
    expand_limit = 3000
    nrof_bch = int(ratio*nrof_chs)
    #bchal = np.random.choice(2, length, p=(0.9, 0.1)).reshape(-1, length)
    # proposed CCA case init_n_chs=1
    # PC(t) case init_n_chs=10
    bchal = np.random.choice(2, length*init_n_chs).reshape(-1, length)
    addchal = bchal.copy()
    print("initial_bchal = {}".format(bchal))
    while len(bchal) < nrof_bch:
        addchal = utils.genChallengesBySAC(length, len(addchal), chs=addchal)
        addchal = np.unique(addchal, axis=0)
        bchal = np.concatenate([bchal, addchal])
        bchal = np.unique(bchal, axis=0)
        if len(addchal) > expand_limit:
            sampledIdx = np.random.choice(len(bchal), expand_limit, replace=False)
            addchal = bchal[sampledIdx]
        print("bchal.shape = {}".format(bchal.shape))
    bchal = bchal[:nrof_bch]
    if debug:
        pdb.set_trace()
    if ratio != 1:
        nrof_rand_c = nrof_chs - nrof_bch
        rchal = utils.genNChallenges(length, nrof_rand_c)
        bchal = np.concatenate([bchal, rchal])

    #Shuffle
    if shuffle:
        bchal = chShuffle(bchal)
    return bchal

def generatedPatternVectorByChallengeIdx(length, indices, HDT, PC=False):
    #np.random.seed(42)
    if indices.size == 0:
        indices = np.arange(length)
    indicesCombinations = []
    if HDT > 1:
        indicesCombinations = list(combinations(indices, HDT))
    #Proporgation Criterion, HDT(e,t), for all t < HDT
    if PC:
        for i in range(1, HDT):
            com = list(combinations(indices, i))
            indicesCombinations = indicesCombinations + com

    pvSize = len(indicesCombinations)
    patternVectors = np.zeros((pvSize, length))
    for i, combs in enumerate(indicesCombinations):
        #pdb.set_trace()
        patternVectors[i, np.array(combs,dtype=int)]=1
    return patternVectors

def generatedPatternVectorByRandom(length, HDT, nrof_patterns):
    #np.random.seed(42)
    indices = np.arange(length)
    current_idx = 0
    indicesCombinations = []
    patternVectors = np.zeros((nrof_patterns, length))
    for i in range(1, HDT+1):
        nrof_comb = comb(length, i)
        if current_idx + nrof_comb < nrof_patterns:
            com = list(combinations(indices, i))
            for j, combs in enumerate(com):
                patternVectors[current_idx+j, np.array(combs,dtype=int)]=1
            current_idx = current_idx + nrof_comb
            #pdb.set_trace()
        else:
            remaining_patterns = np.array([])
            previous_idx = current_idx
            while current_idx < nrof_patterns-1:
                for k in range(2*(nrof_patterns-current_idx)):
                    if len(remaining_patterns)==0:
                        remaining_patterns = np.random.choice(length, i, replace=False).reshape(1, i)
                    else:
                        sampled_idx = np.random.choice(length, i, replace=False).reshape(1, i)
                        remaining_patterns = np.append(remaining_patterns, sampled_idx, axis=0)
                remaining_patterns = np.sort(remaining_patterns, axis=1)
                remaining_patterns = np.unique(remaining_patterns, axis=0)
                remaining_patterns = remaining_patterns[:(nrof_patterns-current_idx)]
                current_idx = previous_idx + len(remaining_patterns)
            for j, combs in enumerate(remaining_patterns):
                patternVectors[previous_idx+j, np.array(combs,dtype=int)]=1
            #pdb.set_trace()
            break
    #print('end_generatedPatternVectorByRandom')
    #pdb.set_trace()
    return patternVectors


def prepareHDTChallenges(length, nrof_queried_challenges, cidx=np.array([]), HDT=1, ratio=0.5, shuffle=True, PC=False, randomSample=False):
    #np.random.seed(42)
    patternSet = np.array([])
    nrof_rnd_challenges = 1000
    nrof_pattern_challenges = int(ratio*nrof_queried_challenges)
    if randomSample:
        nrof_pattern = nrof_queried_challenges//nrof_rnd_challenges
        patternSet = generatedPatternVectorByRandom(length, HDT, nrof_pattern)
    else:
        patternSet = generatedPatternVectorByChallengeIdx(length, cidx, HDT, PC)
        nrof_rnd_challenges = math.ceil(nrof_pattern_challenges/len(patternSet))
    seedChallenges = utils.genNChallenges(length, nrof_rnd_challenges)
    challengeSet = seedChallenges.copy()

    for p in patternSet:
        patternChallenges = np.logical_xor(p, seedChallenges)
        challengeSet = np.concatenate([challengeSet, patternChallenges])
    challengeSet = challengeSet[:nrof_pattern_challenges]
    challengeSet = addRandomChallenges(length, challengeSet, ratio, nrof_pattern_challenges, nrof_queried_challenges)
    #Shuffle
    if shuffle:
        challengeSet = chShuffle(challengeSet)
    #pdb.set_trace()
    return challengeSet

def addRandomChallenges(length, pchs, ratio, nrof_pattern_c, nrof_challenges):
    if ratio != 1:
        nrof_rand_c = nrof_challenges - nrof_pattern_c
        rchal = utils.genNChallenges(length, nrof_rand_c)
        pchs = np.concatenate([pchs, rchal])
        return pchs
    else:
        return pchs

def chShuffle(chs):
    nrof_chs = len(chs)
    chidx = np.arange(nrof_chs)
    np.random.shuffle(chidx)
    return chs[chidx]

def HDDist(chs):
    nrof_chs = chs.shape[0]
    length = chs.shape[1]
    hd_hist = np.zeros(length+1)
    for i in range(nrof_chs):
        hd = np.logical_xor(chs[i], chs[i+1:]).sum(axis=1)
        maxHD = np.amax(hd)
        for j in range(maxHD):
            hd_hist[j] = np.count_nonzero(hd==j)
        if HDDist:
            print("HDDist. i={}".format(i))
    return hd_dist

def HDAvg(hddist):
    nrof_chs = hddist.sum()
    length = hddist.shape[0]
    hdweighted = hddist*np.arange(length+1)
    avgHD = hdweighted.sum()/nrof_chs
    return avgHD

'''
# not working
def prepareChallenges(length, nrof_chs, bias = 0.05):
    chs = utils.genNChallenges(length, nrof_chs//2, b=bias)
    chs_rand = utils.genNChallenges(length, nrof_chs//2)
    chs = np.concatenate([chs, chs_rand])
    chidx = np.arange(nrof_chs)
    np.random.shuffle(chidx)
    chs = chs[chidx]
    if debug:
        pdb.set_trace()
    return chs
'''
