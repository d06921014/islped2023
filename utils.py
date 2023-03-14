import numpy as np
from decimal import *
from functools import reduce
import math
import pdb

debug = True

def genNChallenges(length=8, nrows=1, b=-1):
    #np.random.seed(42)
    c = np.random.choice(2,length*nrows)
    if b != -1:
        c = np.random.choice(2,length*nrows, p=[b, 1-b])
    c = c.reshape(nrows,length)
    return c

def genChallengesBySAC(length, nChallenges, chs=np.array([])):
    if chs.size == 0:
        chs = np.random.choice(2, nChallenges*length).reshape(nChallenges, length)
    hd1 = np.tile(chs, (length+1, 1, 1))
    # for each index in filp array, for each challenge, flip the index-th bit
    pivot = hd1[length]
    for i in range(length):
        hd1[i, :, i] = np.logical_not(pivot[:,i]).astype(int)
    hd1 = hd1.reshape(nChallenges*(length+1), length)
    return hd1

def softResToHard(softres):
    sres = softres.copy()
    sres[sres>0.5]=1
    sres[sres<=0.5]=0
    return sres

def getReliabilityMask(res, threshold):
    if len(res.shape)!=1:
        print("utils-get stable mask: dim of response is unexpected.")
        return
    nrofres = res.shape[0]
    mask = np.zeros(nrofres, dtype='bool')
    mask[res <= threshold] = True
    mask[res >= 1-threshold] = True
    return mask

def diff(res1, res2):
    nrof_res = res1.flatten().shape[0]
    return (res1.flatten()==res2.flatten()).sum()/nrof_res

def getDiffMsk(res1, res2):
    if len(res1)!=len(res2):
        print("LENGTH MISMATCH.")
        return
    msk = (res1.flatten()==res2.flatten())
    print("Totoal: {}, diff(%): {}.".format(len(msk), msk.sum()/len(msk)))
    return msk

'''
def data_partitioning(x, y, thres):
    set_num = len(thres)+1
'''

def append_ndarr(arr, arr_ele):
    if arr is None:
        arr = arr_ele
    else:
        arr = np.concatenate([arr, arr_ele], axis=1)
    return arr

def genChallengesforSAC(length, nChallenges):
    chs = np.random.choice(2, nChallenges*length).reshape(nChallenges, length)
    hd1 = np.tile(chs, (length+1, 1, 1))
    # for each index in filp array, for each challenge, flip the index-th bit
    pivot = hd1[length]
    for i in range(length):
        hd1[i, :, i] = np.logical_not(pivot[:,i]).astype(int)
    #pdb.set_trace()
    hd1 = hd1.reshape(nChallenges*(length+1), length)
    return hd1

# return n apuf delay vector [nrows, nStage+1]
def genDelayVector(mu=0, sigma=1, nStage=8, nrows=1):
    return np.reshape(np.random.normal(mu, sigma, (nStage+1)*nrows), (nrows, nStage+1))

#===================================================================
# Authentication
#===================================================================
def factorial(n): 
    if n < 2: return 1
    return reduce(lambda x, y: x*y, range(2, int(n)+1))

# s: nrof successful trails.
# n: total nrof trails.
# p: probability of a successful trail.
# F(s,n,p) = Pr(X<=s), X~Bin(n, p)
def cum_bin_prob(s, n, p):
    x = 1.0 - p

    a = n - s
    b = s + 1

    c = a + b - 1

    prob = 0.0

    for j in range(a, c + 1):
        prob += factorial(c) / (factorial(j)*factorial(c-j)) \
                * x**j * (1 - x)**(c-j)

    return prob


# auth_thres = [0.5, 1]
def false_rejected(n, p1, p2, auth_thres, ebit=1):
    x = math.floor(auth_thres*n)
    puf1_err = cum_bin_prob(n-ebit-1, n, p1)
    puf2_err = cum_bin_prob(x, n, p2)
    pfr = puf1_err + puf2_err - puf1_err*puf2_err
    #print("p1 = {}, p2 = {}.".format(p1, p2))
    print("puf#1 err prob = {}, puf#2 err prob = {}, P_FR = {}".format(puf1_err, puf2_err, pfr))
    #print("Approx. Auth rate = {}".format(1-pfr))
    return pfr

def false_accept(n, auth_thres, ebit=1):
    k = math.ceil(auth_thres*n)

    getcontext().prec = 30
    prob = Decimal(0)
    total_trails = Decimal(0)

    for i in range(n-ebit, n + 1):
        total_trails += Decimal(factorial(n) / (factorial(i)*factorial(n-i)))

    for j in range(k, n + 1):
        prob += Decimal(factorial(n) / (factorial(j)*factorial(n-j)))
    pa = prob*Decimal(math.pow(0.5, n))
    pr = Decimal(1) - pa
    pfa = Decimal(1.0) - pr**Decimal(total_trails+1)
    #if auth_thres > 0.85:
        #pdb.set_trace()
    print("Authentication thres = {}, total trails = {} ,P_FA = {}".format(auth_thres, total_trails, pfa))

