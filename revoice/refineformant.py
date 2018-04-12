from .common import *
from . import instantfrequency

def calcError(F, bw, amp, hFreq, hAmp, sr):
    vtAmp = calcKlattFilterBankResponseMagnitude(hFreq, F, bw, amp, sr)
    if((vtAmp <= 0.0).any()):
        return None
    err = calcItakuraSaitoDistance(hAmp, vtAmp)
    return err

def findBestAmp(F, bw, hFreq, hAmp, sr, costMode = False):
    minError = np.inf
    bestAmp = None
    ampIpl = ipl.interp1d(np.concatenate(((0,), hFreq)), np.concatenate(((hAmp[0],), hAmp)), kind = "linear", bounds_error = False, fill_value = hAmp[-1])
    for FFac in np.linspace(0.5, 1.5, 9):
        amp = ampIpl(F * FFac)
        err = calcError(F, bw, amp, hFreq, hAmp, sr)
        if(err is None):
            continue
        if(err < minError):
            minError = err
            bestAmp = amp
    if(costMode):
        return minError
    return minError, bestAmp

def findBestBwAmp(F, refBw, hFreq, hAmp, sr, costMode = False):
    minError = np.inf
    bestBw = None
    bestAmp = None
    amp = None
    for bwFac in np.linspace(0.5, 1.5, 9):
        bw = refBw * bwFac
        ret = findBestAmp(F, bw, hFreq, hAmp, sr, costMode = costMode)
        if(costMode):
            err = ret
        else:
            err, amp = ret
        if(err is None):
            continue
        if(err < minError):
            minError = err
            bestAmp = amp
            bestBw = bw 
    if(costMode):
        return minError
    return minError, bestBw, bestAmp

class Processor:
    def __init__(self, sr, **kwargs):
        self.samprate = sr
    
    def __call__(self, hFreqList, hAmpList, FList, bwList):
        (nHop, _) = FList.shape
        nyq = self.samprate * 0.5

        refinedFList = np.zeros(FList.shape)
        refinedBwList = np.zeros(bwList.shape)
        refinedAmpList = np.zeros(FList.shape)

        for iHop in range(nHop):
            F = FList[iHop]
            bw = bwList[iHop]
            need = np.logical_and(F > 0, F < nyq)
            F = F[need]
            bw = bw[need]
            
            hFreq = hFreqList[iHop]
            hAmp = hAmpList[iHop]
            need = np.logical_and(hFreq > 0, hFreq < nyq)
            hFreq = hFreq[need]
            hAmp = hAmp[need]

            refinedFList[iHop, need] = so.fmin(findBestBwAmp, F, args = (bw, hFreq, hAmp, self.samprate, True)).xopt
            _, refinedBwList[iHop, need], refinedAmpList[iHop, need] = findBestBwAmp(refinedFList[iHop], bw, hFreq, hAmp, self.samprate)
            
        return refinedFList, refinedBwList, refinedAmpList



