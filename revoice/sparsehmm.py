import sys
import numpy as np
from .common import *

@nb.jit(fastmath=True, nopython = True, cache = True)
def hmmViterbiForwardCore(obs, oldDelta, sourceState, targetState, stateTransProb):
    nTrans = stateTransProb.shape[0]
    nState = oldDelta.shape[0]
    assert obs.shape == oldDelta.shape
    
    delta = np.zeros(nState, dtype=np.float32)
    psi = np.zeros(nState, dtype=np.int32)
    currValue = oldDelta[sourceState] * stateTransProb

    for iTrans in range(nTrans):
        ts = targetState[iTrans]
        if currValue[iTrans] > delta[ts]:
            delta[ts] = currValue[iTrans] # will be multiplied by the right obs later
            psi[ts] = sourceState[iTrans]
    delta *= obs

    return delta, psi

class ViterbiDecoder:
    def __init__(self, nState, nTrans):
        assert nState > 0 and nTrans > 0 and nTrans <= nState * nState
        self.nState, self.nTrans = nState, nTrans

        self.oldDelta = None
        self.psi = None # without first frame
        self.usedPsi = None
    
    def initialize(self, firstObsProb, initStateProb):
        nState = self.nState
        assert firstObsProb.shape == (nState,)
        assert initStateProb.shape == (nState,)

        # init first frame
        oldDelta = initStateProb * firstObsProb
        deltaSum = np.sum(oldDelta)
        if(deltaSum > 0.0):
            oldDelta /= deltaSum
        
        self.oldDelta = oldDelta
        self.psi = np.zeros((0, nState), dtype = np.int32)
        self.usedPsi = 0
    
    def preserve(self, nFrame):
        self.psi = np.concatenate((self.psi, np.zeros((nFrame, self.nState), dtype = np.int32)))
    
    def feed(self, obsStateProbList, sourceState, targetState, stateTransProb):
        nState, nTrans = self.nState, self.nTrans
        if(obsStateProbList.ndim == 1):
            obsStateProbList = obsStateProbList.reshape(1, nState)

        assert self.oldDelta is not None
        assert obsStateProbList.shape[1:] == (nState,)
        assert sourceState.shape == (nTrans,)
        assert targetState.shape == (nTrans,)
        assert stateTransProb.shape == (nTrans,)

        nFrame = obsStateProbList.shape[0]
        if self.usedPsi + nFrame > self.psi.shape[0]:
            self.psi = np.resize(self.psi, (self.usedPsi + nFrame + 16, nState))
        psi = self.psi[self.usedPsi:self.usedPsi + nFrame]
        oldDelta = self.oldDelta
        # rest of forward step
        for iFrame in range(nFrame):
            delta, psi[iFrame] = hmmViterbiForwardCore(obsStateProbList[iFrame], oldDelta, sourceState, targetState, stateTransProb)
            deltaSum = np.sum(delta)

            if(deltaSum > 0.0):
                oldDelta = delta / deltaSum
                #scale[iFrame] = 1.0 / deltaSum
            else:
                print("WARNING: Viterbi decoder has been fed some invalid probabilities.", file = sys.stderr)
                oldDelta.fill(1.0 / nState)
                #scale[iFrame] = 1.0
        self.oldDelta = oldDelta
        self.usedPsi += nFrame
    
    def finalize(self):
        self.oldDelta = None
        self.psi = None

    def readDecodedPath(self):
        oldDelta = self.oldDelta
        nFrame = self.usedPsi + 1
        psi = self.psi[:self.usedPsi]

        # init backward step
        bestStateIdx = np.argmax(oldDelta)

        path = np.ndarray(nFrame, dtype=np.int32) # the final output path
        path[-1] = bestStateIdx

        # rest of backward step
        for iFrame in reversed(range(nFrame - 1)):
            path[iFrame] = psi[iFrame][path[iFrame + 1]] # psi[iFrame] is iFrame + 1 of `real` psi
        return path