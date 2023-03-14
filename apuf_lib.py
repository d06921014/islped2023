import numpy as np
import utils
from utils import genDelayVector
import pdb

debug = False

# return n for n-xor puf challenges [nrows, length]
#def xorpufGenchallenge(length=8, nxor=1, nrows=1):
#    c = np.random.choice(2,length*nrows).reshape(nrows, length)
#    #c[c==0]=-1
#    c = np.tile(c, nxor).reshape(nrows, nxor,length)
#    return c

# return n challenge vectors that encoded challenges [nrows, length+1]

def challengeTransform(achallenge, chalSize, nrows=1):
    aphi = np.ones([nrows,chalSize+1])
    for i in range(nrows):
        for j in range(chalSize):
            if np.logical_xor.reduce(achallenge[i,j:]):
                aphi[i,j] = -1
            else:
                aphi[i,j] = 1
    return aphi
'''
def xorChallengeTransform(achallenge, chalSize, nxor, nrows=1):
    aphi = challengeTransform(achallenge, chalSize, nrows)
    aphi = np.tile(aphi, nxor).reshape(nrows, nxor, chalSize+1)
    return aphi
'''

# challenge: parity vector
class CRP:
    def __init__(self, challenges, response, raw = np.array([])):
        self.challenge = challenges
        self.response = response
        self.ch_raw = raw
        if len(challenges) != len(response):
            print("invalid dim: challenge shape={}, response shape = {}".format(challenges.shape, response.shape))

#==========================================================================
# ignore the above functions if read generated challenges from csv directly

# mode change mechanism design
# divide into div=4 part, get the result by xor the first bit from each part
# expected challenge:[0,1]
#============================================================================

class XORAPUF:
    def __init__(self, nStage, nxor, envNSigma = 1.5, delays = np.array([])):
        self.nstage = nStage
        self.nxor = nxor
        self.apufs = delays
        self.evNSig = envNSigma
        if not delays.any():
            self.apufs = genDelayVector(nStage = nStage, nrows=nxor)
        if len(self.apufs.shape) < 2:
            print("XORPUF: invalid length for APUFs: {} ".format(len(self.apufs.shape)))
        if len(self.apufs) != nxor:
            print("XORPUF: number of APUFs should be {}".format(nxor))

    def getPufResponse(self, aphis, nchs=1, isParity=True, noisefree = False):
        if not isParity:
            aphis = challengeTransform(aphis, aphis.shape[1], nrows = nchs)
        if len(aphis.shape) == 2:
            aphis = np.tile(aphis, self.nxor).reshape(nchs, self.nxor, self.nstage+1)
        XORAResponse = np.ones((nchs))
        individual_apsReses = np.ones((nchs, self.nxor))
        for i in range(nchs):
            #pdb.set_trace()
            culDelay = np.sum(self.apufs*aphis[i],axis=1)
            if not noisefree:
                noise = np.random.normal(0, self.evNSig, self.nxor)
                culDelay = culDelay + noise
            #print("culDelay, before sign\n {}".format(culDelay))
            culDelay[culDelay > 0] = 1
            culDelay[culDelay <= 0] = 0
            individual_apsReses[i] = culDelay
            XORAResponse[i] = np.logical_xor.reduce(culDelay).astype(int)
            if debug:
                #pdb.set_trace()
                print("cul_delay (after sign):{}, \nXORAResponse[{}]=\n{}".format(culDelay, i, XORAResponse[i]))
        return XORAResponse, individual_apsReses

    def setNSigma(self, nsigma):
        self.evNSig = nsigma

    # Save delay vector and noise
    def savePUFInstance(self, path):
        np.save(path, self.apufs)
        print("save instance - path={}".format(path))
        return


class APUF:
    def __init__(self, nStage, mu=0, sigma=1, nSigma = 0.5, delay = np.array([])):
        self.nstage = nStage
        self.delayVec = delay
        if not delay.any():
            self.delayVec = genDelayVector(mu = mu, sigma = sigma, nStage = nStage)
        self.nsigma = nSigma

    # Expected input is a parity vector.
    # return 1-bit responses of one apuf, response.shape = [nchallenges, 1] 
    def getPufResponse(self, challenge, nchs=1, isParity=True, noisefree = False):
        if not isParity:
            challenge = challengeTransform(challenge, challenge.shape[1], nrows = nchs)
        nFreeResponse = self.rawDelay(challenge)
        noise = np.random.normal(0, self.nsigma, nchs)

        AResponse = nFreeResponse + noise
        if noisefree:
            AResponse = nFreeResponse.copy()
        nFreeResponse[nFreeResponse > 0] = 0
        nFreeResponse[nFreeResponse < 0] = 1
        AResponse[AResponse > 0] = 0
        AResponse[AResponse < 0] = 1

        #print("Reliability = {} %".format((np.sum(AResponse==nFreeResponse)/nchs)*100))
        return AResponse

    def rawDelay(self, parity):
        return np.add.reduce(self.delayVec*parity, 1)

    def setNSigma(self, nsigma):
        self.nsigma = nsigma

    # Save delay vector and noise
    def savePUFInstance(self, path):
        tmp = np.concatenate([self.delayVec.flatten(), np.array([self.nsigma])])
        np.save(path, tmp)
        print("save instance - path={}".format(path))
        return

class CSAPUF:
    def __init__(self, apuf1, apuf2, dim, nrof_row):
        if nrof_row < dim:
            print("CSAPUF build failed: m cannot smaller than n.")
            return
        self.n = dim
        self.m = nrof_row
        self.puf_secret = apuf1
        self.puf_error = apuf2

        mask = np.zeros(self.m)
        mask[:self.n]=1
        np.random.shuffle(mask)
        self.trapdoor = mask.astype(bool)
        
    def getPufResponse(self, challenge, isParity=False, noisefree = False):
        if self.m != challenge.shape[0]:
            print("CSAPUF - require {} challenges per batch".format(self.m))
            return
        schal = challenge[self.trapdoor]
        secret = self.puf_secret.getPufResponse(schal, self.n, isParity, noisefree)
        error = self.puf_error.getPufResponse(challenge, self.m, isParity, noisefree)
        response = (challenge.dot(secret)+error)%2
        '''
        # for debug
        for i in range(self.m):
            ads = np.logical_and(challenge[i], secret)
            adsp = np.logical_xor.reduce(ads)
            adspe = np.logical_xor(adsp, error[i])
            #pdb.set_trace()
            if adspe != response[i]:
                print("csapuf getresponse {} - answer is not correct!".format(i))
        '''
        return response, secret, error

    # Save delay vector and noise
    def savePUFInstance(self, path):
        pufIns = np.concatenate([[self.puf_secret], [self.puf_error]])
        np.save(path, tmp)
        print("save instance - path={}".format(path))
        return

    def initializeIPUF(self, path, nsig):
        params = np.load(path)
        self.n = len(params[0])
        self.m = len(params[1])
        self.puf_secret = APUF(stage = self.n, delay = params[0], nSigma = nsig)
        self.puf_error = APUF(stage = self.n, delay = params[1], nSigma = nsig)
        print("Initialize CSAPUF instance - path={}".format(path))
        return

class IPUF:
    def __init__(self, x, y, dim, index=-1):
        if dim <= 0 or x.nstage != dim or y.nstage != dim+1:
            print("IPUF - Invalid length = {}".format(dim))
        self.idx = index
        if index < 0:
            self.idx = dim//2
        self.xpuf = x
        self.ypuf = y
    # Raw challenge(s) as input
    def getPufResponse(self, chals, noisefree = True):
        nchs = chals.shape[0]
        #xchals = chals.copy()

        aphisx = challengeTransform(chals, chals.shape[1], nrows = nchs)
        xres, _ = self.xpuf.getPufResponse(aphisx, nchs=nchs, noisefree = noisefree)
        xchals = np.insert(chals, self.idx, xres, axis=1)
        aphisy = challengeTransform(xchals, xchals.shape[1], nrows = nchs)
        yres, _ = self.ypuf.getPufResponse(aphisy, nchs=nchs, noisefree = noisefree)
        
        return xres, yres
    def savePUFInstance(self, path):
        pufIns = np.concatenate([[self.xpuf], [self.ypuf]])
        np.save(path, tmp)
        print("save instance - path={}".format(path))
        return
        
    def initializeIPUF(self, path ,index=-1):
        params = np.load(path)
        self.xpuf = params[0]
        self.ypuf = params[1]
        self.dim = xpuf.shape[1]
        if index < 0:
            self.idx = dim//2
        print("Initialize IPUF instance - path={}".format(path))
        return

# Ex: 9-bit DCH 64-APUF, LFSR_poly=[9, 8, 4, 1], 2*(8:1)-ff-MUX, 1*(8:1)-sx-MUX, 7 inter-arbiter
#     DCH(length, nxor=1, nsxconfig=3, lfsrSize=9, lfsrpoly=[9, 8, 4, 1])
#
# Ex: 11-bit DCH 64-2XOR-APUF, LFSR_poly=[11, 2], 4*(4:1)-ff-MUX, 1*(8:1)-sx-MUX, 6 inner-arbiters. (2(XOR)*3) 
#     DCH(length, nxor=2, nsxconfig=3 lfsrSize=11, lfsrpoly=[11, 2])
#
# the size of LFSR are divided equally (by the number of MUXes) to determine the number of branches
# Ex: 11-bit DCH 2XOR 1-SX PUF. 4 (4:1) MUXes for ff branch, 1 (8:1) MUX for self-xor
class DCH:
    def __init__(self, nStage, nxor=2, nsxconfig=3, nff=2, lfsrSize=11, lfsrpoly=[11, 2, 1], delay=np.array([]), br_delay=np.array([])):
        from pylfsr import LFSR
        self.nStage = nStage
        self.nxor = nxor
        self.nsxbr = 1
        self.nsxconfig = nsxconfig
        self.nff = nff

        # self.nbrs: # of branches for each APUF path
        self.nbrs = np.power(2, (lfsrSize - self.nsxconfig)//(nff*nxor)) - 1 
        mu = 0
        sigma = 1 # apuf's variations
        self.delayVec = delay
        self.br_delay = br_delay
        if not delay.any():
            self.delayVec = genDelayVector(mu = mu, sigma = sigma, nStage = nStage, nrows=nxor)
        if len(self.delayVec) != nxor:
            print("Number of Delay Vectors mismatch. required: {}, found: {}".format((nxor, nStage), self.delayVec.shape))
        if not br_delay.any():
            self.br_delay = np.random.normal(mu, sigma/np.sqrt(self.nStage + 1), (self.nxor, self.nbrs))

        self.lfsr = LFSR(initstate=np.random.randint(2, size=(lfsrSize)).tolist(), fpoly=lfsrpoly)

        # FeedForward Path
        ffrange = np.arange(nStage//2, nStage-2)
        self.ffidx = self.getConfigIdx(ffrange, self.nff, self.nxor)

        branchrange = np.arange(3, nStage//2)
        # Indices of APUF's branches. 
        # Shape=(nxor, narb)      
        self.branchidx = self.getConfigIdx(branchrange, self.nbrs, self.nxor)
        self.nSigma = 1.5 # for noise
        if debug:
            print("Constructing an {}-bit {}-sx DCH with {}-bit {}XOR/APUF, {}-({}:1)-MUX and {}-arbiter branches".format(
                     self.lfsr.M, self.nsxbr , self.nStage, self.nxor\
                     ,self.nxor*self.nff, 2**(self.nff), len(self.branchidx.flatten())))
            pdb.set_trace()

    def getConfigIdx(self, idxRange, size, nrows):
        #pdb.set_trace()
        return np.array([np.sort(np.random.choice(idxRange, size=size, replace=False)) for i in range(nrows)])

    def getPufResponse(self, chals, noisefree = False):
        res = None
        for i in range(2**self.lfsr.M-1):
            if res is None:
                res = self.getOneStatePufResponse(chals, noisefree)
            else:
                one_state_res = self.getOneStatePufResponse(chals, noisefree)
                res = np.concatenate([res, one_state_res])
            lfsr_out = self.lfsr.next()
        return res


    # All challeges are shared the same LFSR state
    def getOneStatePufResponse(self, chal, noisefree = False):
        # Assume noise for each stage is iid zero-mean gaussian dist.
        # Each arbiter branch: mu=(idx/n)mu, var = (idx/n)sigma^2
        nchs = chal.shape[0]
        parity = challengeTransform(chal, chal.shape[1], nrows = nchs)
        parity_xor = np.tile(parity, self.nxor).reshape(nchs, self.nxor, self.nStage+1)
        raw_delay = self.delayVec*parity_xor
        ap_noise = np.random.normal(0, self.nSigma/np.sqrt(self.nStage + 1), size=(nchs, self.nxor, self.nStage + 1))

        # Response of each branch
        # Shape: (nchs, nxor, n_ff_mux fanin)
        br_res = self.getAllBranchRes(nchs, raw_delay, ap_noise, noisefree)

        # Init last value of mux
        # Last value should be the challenge bit of the ff index
        # Shape of ff_mux: (nchs, nxor, n_ff_mux, n_ff_mux fanin)
        ff_mux_fanin = np.concatenate([br_res, np.zeros((nchs, self.nxor, 1))], axis=2)
        ff_mux = ff_mux_fanin.repeat(self.nff, axis=1).reshape(nchs, self.nxor, self.nff, self.nbrs+1)

        sx_mux_fanin = br_res.reshape(nchs, self.nbrs*self.nxor)
        nrof_sx_mux_fanin = sx_mux_fanin.shape[1]
        # Padding, EX: suppose sx_mux has 6 bits, 2^[ceil(log2(6))] - 6 = 2 bit
        sx_mux_pad = int(np.power(2, np.ceil(np.log2(nrof_sx_mux_fanin))) - nrof_sx_mux_fanin)
        sx_mux = np.concatenate([br_res.reshape(nchs, -1), np.zeros((nchs, sx_mux_pad))], axis=1)

        for i in range(self.nxor):
            for j in range(self.nff):
                ff_mux[:, i, j, -1] = chal[:, self.ffidx[i, j]]
        if debug:
            print("mux_input_array =\n{}".format(ff_mux))
            #pdb.set_trace()

        # Translate Configurations From the LFSR State
        # All challenges shared the same mux configuration
        nffCfbits = int(np.ceil(np.log2(self.nbrs)))
        nffMux = self.nxor*self.nff
        conf_arr = np.array([self.lfsr.state[i : i + nffCfbits] if i + nffCfbits <= nffMux*nffCfbits \
                                                              else self.lfsr.state[i:] \
                                                              for i in range(0, nffMux*nffCfbits+1, nffCfbits)])
        mux_select = np.array([0]*len(conf_arr))
        # element of the conf_arr must be "int"
        for i in range(len(conf_arr)):
            mux_select[i] = int(''.join(str(e) for e in conf_arr[i]), 2)
        if debug:
            print("lfsr_state = \n{}\n".format(self.lfsr.state)\
                   +"mux_config = \n{}\n".format(conf_arr)\
                   +"mux_selector = \n{}".format(mux_select))
            #pdb.set_trace()

        # Replace challenge bits in corresponding feedforward stages
        ff_chal = np.expand_dims(chal, axis=1).repeat(self.nxor, axis=1)
        ff_mux_select = mux_select[:self.nxor*self.nff].reshape(self.nxor, self.nff)
        for i in range(self.nxor):
            for j in range(self.nff):
                idx = self.ffidx[i, j]
                ff_chal[:, i, idx] = ff_mux[:, i, j, ff_mux_select[i, j]]

        # Apply New Challenge to the Original APUF
        tmp = ff_chal.reshape(-1, ff_chal.shape[-1])
        ff_parity = challengeTransform(tmp, tmp.shape[1], nrows = tmp.shape[0]).reshape(nchs, self.nxor, -1)
        ff_raw_delay = np.add.reduce(self.delayVec*ff_parity, 2)  
        if noisefree:
            ap_noise.fill(0)
        ap_raw_delay = ff_raw_delay + ap_noise.sum(axis=2)
        apres = ap_raw_delay.copy()
        apres[apres >= 0] = 0
        apres[apres < 0] = 1
        xorapres = np.logical_xor.reduce(apres, axis=1)

        # XORing ffAPUF's response and selected branch
        xor_branch = sx_mux[:, mux_select[-1]]
        res = np.logical_xor(xorapres, xor_branch)
        if debug:
            print("apuf_res =\n{}\n".format(apres)\
                + "mux_xor_branch = \n{}\n".format(xor_branch))
            #pdb.set_trace()
        return res

    def getAllBranchRes(self, nchs, raw_delay, ap_noise, noisefree):
        br_cul_delay = np.zeros((nchs, self.nxor, self.nbrs))
        br_cul_noise = np.zeros((nchs, self.nxor, self.nbrs))
        # br_raw_delay:(chs, 1). br_noise:(chs, 1)
        for i in range(nchs):
            for j in range(self.nxor):
                for k in range(self.nbrs):
                    idx = self.branchidx[j, k]
                    br_cul_delay[i, j, k] = raw_delay[i, j, :idx].sum() + self.br_delay[j, k]

                    if not noisefree:
                        br_arb_noise = np.random.normal(0, self.nSigma/np.sqrt(self.nStage + 1))
                        br_cul_noise =  ap_noise[i, j, :idx].sum() + br_arb_noise

        br_delay_diff = br_cul_delay + br_cul_noise

        if debug:
            print("apuf_raw_delay =\n{}\n".format(raw_delay)\
                + "branch_idx = \n{}\n".format(self.branchidx)\
                + "branch_arb_delay = \n{}\n".format(self.br_delay)\
                + "branch_cul_delay = \n{}\n".format(br_cul_delay)\
                + "branch_cul_noise = \n{}\n".format(br_cul_noise)\
                + "branch_delay_diff = \n{}".format(br_delay_diff))
            pdb.set_trace()
        br_output = br_delay_diff.copy()
        br_output[br_output >= 0]=0
        br_output[br_output < 0]=1
        return br_output

    def setNSigma(self, nsigma):
        self.nSigma = nsigma
