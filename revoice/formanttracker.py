from .common import *
from . import sparsehmm

@nb.jit(nb.float32(nb.float32, nb.float32[:], nb.float32[:]), nopython = True, cache = True)
def _calcStateTransProb(octaveTransCost, srcF, targetF):
    if((srcF <= 0.0).any() or (targetF <= 0.0).any()):
        return eps
    return 1 / np.mean(np.exp(octaveTransCost * np.abs(np.log2(srcF / targetF)))) + eps

@nb.jit(nb.float32[:](nb.int32, nb.float32, nb.int32[:], nb.float32[:, :]), nopython = True, cache = True)
def _calcTransProbList(iHop, octaveTransCost, candFrameOffsetList, FList):
    (nHop, _) = FList.shape
    (nState,) = candFrameOffsetList.shape

    stateTransProbList = np.zeros(nState * nState, dtype=np.float32)
    for iSource in range(nState):
        sourceFrameOffset = candFrameOffsetList[iSource]
        iSourceHop = iHop + sourceFrameOffset
        if(iSourceHop < 0 or iSourceHop >= nHop):
            stateTransProbList[iSource * nState:(iSource + 1) * nState] = eps
            continue
        for iTarget in range(nState):
            targetFrameOffset = candFrameOffsetList[iTarget]
            iTargetHop = iHop + targetFrameOffset
            if(iTargetHop < 0 or iTargetHop >= nHop):
                stateTransProbList[iSource * nState + iTarget] = eps
            else:
                stateTransProbList[iSource * nState + iTarget] = _calcStateTransProb(octaveTransCost, FList[iSourceHop], FList[iTargetHop])
    stateTransProbList /= np.sum(stateTransProbList)
    return stateTransProbList

class Analyzer:
    def __init__(self, nTrack, sr, **kwargs):
        self.nTrack = nTrack
        self.samprate = sr

        self.refFormantFreqList = kwargs.get("refFormantFreqList", formantFreq(np.arange(1, nTrack + 1), 0.15))
        self.deltaFrequencyCost = kwargs.get("deltaFrequencyCost", 1.0)
        self.bandwidthFrequencyCost = kwargs.get("bandwidthFrequencyCost", 1.0)
        self.octaveTransCost = kwargs.get("octaveTransCost", 0.1)

        self.candFrameOffsetList = np.sort(np.asarray(kwargs.get("candOffsetList", np.arange(-16, 17))))

    def calcError(self, hFreq, hAmp, F, bw):
        minError = np.inf
        for bwFac in np.linspace(0.5, 5.0, 17):
            Famp = ipl.interp1d(np.concatenate(((0,), hFreq)), np.concatenate(((hAmp[0],), hAmp)), kind = "linear", bounds_error = False, fill_value = hAmp[-1])(F)
            vtAmp = calcKlattFilterBankResponseMagnitude(hFreq, F, np.clip(bw * bwFac, 50.0, 500.0), Famp.astype(np.float32), self.samprate)
            if((vtAmp <= 0.0).any()):
                continue
            err = calcItakuraSaitoDistance(hAmp, vtAmp)
            if(err < minError):
                minError = err
        return np.exp(minError) ** 2

    def calcSingleStateProb(self, hFreq, hAmp, F, bw):
        if((F <= 0.0).any()):
            return eps
        error = self.calcError(hFreq, hAmp, F, bw)
        return 1 / error + eps

    def calcStateProb(self, iHop, hFreq, hAmp, FList, bwList):
        assert FList.shape == bwList.shape

        (nState,) = self.candFrameOffsetList.shape
        nHop = FList.shape[0]
        
        need = np.logical_and(hFreq > 0, hFreq < self.samprate * 0.5)
        if(np.sum(need) == 0):
            return np.full(nState, eps)
        hFreq = hFreq[need]
        hAmp = hAmp[need]

        obsProbList = np.zeros(nState, dtype=np.float32)
        for iState, frameOffset in enumerate(self.candFrameOffsetList):
            iOffsetHop = frameOffset + iHop
            if(iOffsetHop < 0 or iOffsetHop >= nHop):
                obsProbList[iState] = eps
                continue
            obsProbList[iState] = self.calcSingleStateProb(hFreq, hAmp, FList[iOffsetHop], bwList[iOffsetHop])
        obsProbList /= np.sum(obsProbList)

        return obsProbList

    def __call__(self, hFreqList, hAmpList, FList, bwList):
        assert FList.shape == bwList.shape
        (nHop, _) = FList.shape
        nTrack = self.nTrack

        (nState,) = self.candFrameOffsetList.shape
        assert nState < 32768
        model = sparsehmm.ViterbiDecoder(nState, nState * nState)

        initStateProb = np.full(nState, 1 / nState, dtype=np.float32)
        model.initialize(self.calcStateProb(0, hFreqList[0], hAmpList[0], FList, bwList), initStateProb)
        del initStateProb

        model.preserve(nHop - 1)
        sourceStateList = np.repeat(np.arange(nState, dtype=np.int32), nState)
        targetStateList = np.tile(np.arange(nState, dtype=np.int32), nState)
        for iHop in range(1, nHop):
            print(iHop, nHop)
            stateTransProbList = _calcTransProbList(iHop, self.octaveTransCost, self.candFrameOffsetList, FList)
            model.feed(self.calcStateProb(iHop, hFreqList[iHop], hAmpList[iHop], FList, bwList), sourceStateList, targetStateList, stateTransProbList)
        
        trackedStateList = model.readDecodedPath()
        trackedFList = np.zeros((nHop, nTrack), dtype=np.float32)
        trackedBwList = np.zeros((nHop, nTrack), dtype=np.float32)

        for iHop, iState in enumerate(trackedStateList):
            frameOffset = self.candFrameOffsetList[iState]
            iOffsetHop = frameOffset + iHop
            if(iOffsetHop < 0 or iOffsetHop >= nHop):
                iOffsetHop = iHop

            trackedFList[iHop] = FList[iOffsetHop]
            trackedBwList[iHop] = bwList[iOffsetHop]
        
        return trackedFList, trackedBwList