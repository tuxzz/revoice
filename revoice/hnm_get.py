from .common import *
from . import hnm

@nb.jit(parallel=True, fastmath=True, nopython=True)
def _hnmGetCore(x, f0List, hopSize, B, mvf, sr, removeDC, outHFreqList, outHAmpList, outHPhaseList):
    hFreqList, hAmpList, hPhaseList = outHFreqList, outHAmpList, outHPhaseList
    assert hFreqList.shape == hAmpList.shape == hPhaseList.shape
    assert f0List.shape == (hFreqList.shape[0],)
    (nHop, maxHar) = hFreqList.shape

    for iHop in nb.prange(nHop):
        f0 = f0List[iHop]
        if f0 <= 0:
            hFreqList[iHop] = 0
            continue
        iCenter = int(round(iHop * hopSize))
    
        nHar = min(maxHar, int(mvf / f0))
        hFreq = np.arange(1, nHar + 1) * f0

        windowSize = int(sr / f0 * B * 2.0)
        if windowSize % 2 != 0:
            windowSize += 1
        window = blackman(windowSize)
        windowNormFac = 2.0 / np.sum(window)
        frame = getFrame(x, iCenter, windowSize)
        if removeDC:
            frame = removeDCSimple(frame)
        frame *= window

        hSpec = calcSpectrumAtFreq(frame, hFreq, sr)
        hAmp = np.abs(hSpec) * windowNormFac
        hPhase = np.angle(hSpec)

        nHar = hFreq.shape[0]
        hFreqList[iHop, :nHar] = hFreq
        hAmpList[iHop, :nHar] = hAmp
        hPhaseList[iHop, :nHar] = hPhase

class Analyzer:
    def __init__(self, hopSize, mvf, sr, **kwargs):
        self.samprate = sr
        self.hopSize = hopSize
        self.windowLengthFac = kwargs.get("windowLengthFac", 1.0)
        self.removeDC = kwargs.get("removeDC", True)

        self.mvf = mvf

        assert(self.mvf <= self.samprate / 2)

    def __call__(self, x, f0List, maxHar):
        # constant
        (nX,) = x.shape
        nHop = getNFrame(nX, self.hopSize)
        _, B, _ = getWindow("blackman")
        B *= self.windowLengthFac

        # check input
        assert (nHop,) == f0List.shape
    
        hFreqList = np.zeros((nHop, maxHar), dtype=np.float32)
        hAmpList = np.zeros((nHop, maxHar), dtype=np.float32)
        hPhaseList = np.zeros((nHop, maxHar), dtype=np.float32)
        _hnmGetCore(x, f0List, self.hopSize, B, self.mvf, self.samprate, self.removeDC, hFreqList, hAmpList, hPhaseList)
        return hFreqList, hAmpList, hPhaseList