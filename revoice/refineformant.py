from .common import *
from . import instantfrequency

def calcError(F, bw, amp, hFreq, hAmp, sr):
    vtAmp = calcKlattFilterBankResponseMagnitude(hFreq, F, bw, amp, sr)
    vtAmp = np.clip(vtAmp, eps, np.inf)
    err = calcItakuraSaitoDistance(hAmp, vtAmp)
    return err

def findBestAmp(F, bw, hFreq, hAmp, sr, costMode = False):
    minError = np.inf
    bestAmp = None
    ampIpl = ipl.interp1d(np.concatenate(((0,), hFreq)), np.concatenate(((hAmp[0],), hAmp)), kind = "linear", bounds_error = False, fill_value = hAmp[-1])

    for FFac in np.linspace(0.5, 1.5, 9):
        amp = ampIpl(F * FFac)
        for ampFac in np.linspace(0.5, 1.5, 9):
            err = calcError(F, bw, amp * ampFac, hFreq, hAmp, sr)
            if(err is None):
                continue
            if(err < minError):
                minError = err
                bestAmp = amp
    if(costMode):
        return minError
    return minError, bestAmp

def findBestBwAmp(F, refBw, hFreq, hAmp, sr, costMode = False):
    (nFormant,) = F.shape

    ampIpl = ipl.interp1d(np.concatenate(((0,), hFreq)), np.concatenate(((hAmp[0],), hAmp)), kind = "linear", bounds_error = False, fill_value = hAmp[-1])
    amp = ampIpl(F)
    bw = np.clip(refBw, 50.0, 400.0)

    for iIter in range(24):
        for iFormant in range(nFormant):
            minErr = np.inf
            bestBw = bw[iFormant]
            for bwFac in np.linspace(0.5, 4.0, 11):
                localBw = bw.copy()
                localBw[iFormant] = max(50.0, min(localBw[iFormant] * bwFac, 500.0))
                err = calcError(F, localBw, amp, hFreq, hAmp, sr)
                if(err < minErr):
                    minErr = err
                    bestBw = bw[iFormant] * bwFac
            bw[iFormant] = bestBw
        for iFormant in range(nFormant):
            minErr = np.inf
            bestAmp = amp[iFormant]
            for ampFac in np.linspace(0.5, 1.5, 11):
                localAmp = amp.copy()
                localAmp[iFormant] = max(eps, localAmp[iFormant] * ampFac)
                err = calcError(F, bw, localAmp, hFreq, hAmp, sr)
                if(err < minErr):
                    minErr = err
                    bestAmp = amp[iFormant] * ampFac
            amp[iFormant] = bestAmp
    
    return bw, amp

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
            if(F[0] <= 0):
                continue
            fNeed = np.logical_and(F > 0, F < nyq)
            F = F[fNeed]
            bw = bw[fNeed]
            
            hFreq = hFreqList[iHop]
            if(hFreq[0] <= 0):
                continue
            hAmp = hAmpList[iHop]
            need = np.logical_and(hFreq > 0, hFreq < nyq)
            hFreq = hFreq[need]
            hAmp = hAmp[need]

            bestBw, bestAmp = findBestBwAmp(F, bw, hFreq, hAmp, self.samprate)
            refinedFList[iHop][fNeed] = F
            refinedBwList[iHop][fNeed] = bestBw
            refinedAmpList[iHop][fNeed] = bestAmp

        return refinedFList, refinedBwList, refinedAmpList



