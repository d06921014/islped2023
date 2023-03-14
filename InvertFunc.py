import numpy as np
from decimal import *
import math
import pdb

debug = False

class InvertFunctions:

    def __init__(self, challenges):
        self.challenges = challenges
        self.length = challenges.shape[-1]

    # idx: DNF style input
    # e.g. idx = [[c1, c2], [c3, c4]]] means c1*c2 + c3*c4
    def ANDOR(self, idx):
        if len(idx.shape) != 2:
            nrof_clause = np.ceil(len(idx)/4)
            idx = np.array_split(idx, nrof_clause)
            print("[ANDOR] split indice to DNF. Index:{}".format(idx))

        #tmp = np.array([np.logical_xor.reduce(self.challenge[:, idx[i]], axis=1) for i in range(len(idx))]).T    
        tmp = []
        for pair in idx:
            a = np.logical_and.reduce(self.challenges[:, pair], axis=1)
            tmp.append(a)
        inv = np.logical_or.reduce(np.array(tmp).T, axis=1)
        return inv

    def XOR(self, idx):
        return np.logical_xor.reduce(self.challenges[:, idx], axis=1)

    def AND(self, idx):
        return np.logical_or.reduce(self.challenges[:, idx], axis=1)

    def TFF_AND(self, idx, seqLen):
        nrof_idx = len(idx)
        #avgSeqLen = int(math.pow(2, nrof_idx))
        chs = self.challenges.copy()

        if len(chs) % seqLen != 0:
           print("[TFF_AND] Invalid size of challenges: challenges.shape={}".format(self.challenges.shape))
           return
        else:
           chs = chs.reshape(-1, seqLen, self.length)
           inv = np.zeros((len(chs), seqLen))
           # each batch
           for i in range(len(chs)):
               bch = chs[i]
               t = np.logical_and.reduce(bch[:, idx], axis=1)
               q = np.zeros(len(t))
               for j in range(len(t)):
                   if j < len(t)-1:
                       q[j+1] = np.logical_xor(t[j], q[j])
               q[-1] = q[-2]
               inv[i] = q
               if debug:
                   print("idx={}".format(idx))
                   print("chs[idx]=\n{}".format(bch[:, idx]))
                   print("t={} \nq={}".format(t, q))
           #pdb.set_trace()
           return inv

    def TFF_AND_back(self, idx):
        nrof_idx = len(idx)
        chs = self.challenges.copy()
        nrof_chs = len(chs)

        tff_q = np.zeros(nrof_chs)
        t = np.logical_and.reduce(chs[:, idx], axis=1)
        # each batch
        for i in range(nrof_chs):    
            if i < nrof_chs-1:
                tff_q[i+1] = np.logical_xor(t[i], tff_q[i])
            if debug:
                print("idx={}".format(idx))
                print("chs[idx]=\n{}".format(bch[:, idx]))
                print("t={} \nq={}".format(t, tff_q))
        tff_q[-1] = tff_q[-2]
        print("TFF trigger times: {}, Ratio:{}".format(t.sum(), t.sum()/nrof_chs))
        #pdb.set_trace()
        return tff_q
    # Reduce "len(idx)" to "l" dim. by XOR network.
    # "s": subsets. "s[i]": size of i-th subset.
    # "s.sum()" should be the same as the total length of selected indices "K".
    def XOR_REDUCED(self, idx, l, s = np.array([])):
        groupedIdx = None
        idx.sort()
        if s.sum() != len(idx):
            print("[XOR_REDUCED] Size of subset is not specified. Divide into equal-size subsets by default.")
            s_size = len(idx)//l
            groupedIdx = np.split(idx, [i*s_size for i in range(1, l)])
        else:
            cut = 0
            groupedIdx = np.split(idx, [s[:i].sum() for i in range(1,l+1)])
        trigger = []
        for i in range(len(groupedIdx)):
            t = np.logical_xor.reduce(self.challenges[:, groupedIdx[i]],  axis=1)
            trigger.append(t)
        trigger = np.transpose(np.array(trigger))
        return trigger
        
        
#unit test

def unit_test_invf():
    import utils
    length = 8
    nchals = 32

    idx=np.arange(2)
    challenges = utils.genNChallenges(length, nchals)
    f = InvertFunctions(challenges)
    inv = f.TFF_AND(np.arange(2))
    print(inv)
    pdb.set_trace()

if __name__ == "__main__":

    unit_test_invf()

