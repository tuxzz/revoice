import sys
import numpy as np
from .common import *

@nb.jit(nb.types.Tuple((nb.float64[:], nb.int32[:]))(nb.float64[:], nb.float64[:], nb.int32[:], nb.int32[:], nb.float64[:]), nopython = True, cache = True)
def hmmViterbiForwardCore(obs, oldDelta, sourceState, targetState, stateTransProb):
    nTrans = stateTransProb.shape[0]
    nState = oldDelta.shape[0]
    assert obs.shape == oldDelta.shape
    
    delta = np.zeros(nState, dtype = np.float64)
    psi = np.zeros(nState, dtype = np.int32)
    currValue = oldDelta[sourceState] * stateTransProb

    for iTrans in range(nTrans):
        ts = targetState[iTrans]
        if(currValue[iTrans] > delta[ts]):
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
        self.psi = np.zeros((0, nState), dtype = np.int)
    
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
        psi = np.zeros((nFrame, nState), dtype = np.int)
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
        self.psi = np.concatenate((self.psi, psi))
    
    def finalize(self):
        self.oldDelta = None
        self.psi = None

    def readDecodedPath(self):
        oldDelta = self.oldDelta
        nFrame = self.psi.shape[0] + 1
        psi = self.psi

        # init backward step
        bestStateIdx = np.argmax(oldDelta)

        path = np.ndarray(nFrame, dtype = np.int) # the final output path
        path[-1] = bestStateIdx

        # rest of backward step
        for iFrame in reversed(range(nFrame - 1)):
            path[iFrame] = psi[iFrame][path[iFrame + 1]] # psi[iFrame] is iFrame + 1 of `real` psi
        return path