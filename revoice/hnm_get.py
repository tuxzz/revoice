from .common import *
from . import hnm

class Analyzer:
    def __init__(self, hopSize, mvf, sr, **kwargs):
        self.samprate = sr
        self.hopSize = hopSize
        self.window = kwargs.get("window", "blackman")
        self.windowLengthFac = kwargs.get("windowLengthFac", 1.0)
        self.removeDC = kwargs.get("removeDC", True)

        self.mvf = mvf
        self.gridSearchPointList = kwargs.get("gridSearchPointList", None)#np.linspace(0.75, 1.25, 32))

        assert(self.mvf <= self.samprate / 2)

    def __call__(self, x, f0List, maxHar):
        # constant
        (nX,) = x.shape
        nHop = getNFrame(nX, self.hopSize)
        windowFunc, B, _ = getWindow(self.window)
        B *= self.windowLengthFac

        # check input
        assert (nHop,) == f0List.shape
    
        hFreqList = np.zeros((nHop, maxHar))
        hAmpList = np.zeros((nHop, maxHar))
        hPhaseList = np.zeros((nHop, maxHar))
        for iHop, f0 in enumerate(f0List):
            if(f0 <= 0):
                hFreqList[iHop] = 0
                continue
            iCenter = int(round(iHop * self.hopSize))
            def costFunction(cf0, result = False):
                nHar = min(maxHar, int(self.mvf / cf0))
                hFreq = np.arange(1, nHar + 1) * cf0

                windowSize = int(self.samprate / cf0 * B * 2.0)
                if(windowSize % 2 != 0):
                    windowSize += 1
                window = windowFunc(windowSize)
                windowNormFac = 2.0 / np.sum(window)
                frame = getFrame(x, iCenter, windowSize)
                if(self.removeDC):
                    frame = removeDCSimple(frame)
                frame *= window

                hSpec = calcSpectrumAtFreq(frame, hFreq, self.samprate)
                hAmp = np.abs(hSpec) * windowNormFac
                hPhase = np.angle(hSpec)

                if(result):
                    return hFreq, hAmp, hPhase
                else:
                    t = np.arange(-windowSize // 2, windowSize // 2) / self.samprate
                    return np.mean((frame - hnm.synthSinusoid(t, hFreq, hAmp, hPhase, self.samprate) * window) ** 2)
            
            if(self.gridSearchPointList is not None and len(self.gridSearchPointList) > 0):
                newF0 = minimizeScalar(costFunction, self.gridSearchPointList * f0)
                if(costFunction(newF0) < costFunction(f0)):
                    f0 = newF0
            hFreq, hAmp, hPhase = costFunction(f0, result = True)
            nHar = hFreq.shape[0]
            hFreqList[iHop, :nHar] = hFreq
            hAmpList[iHop, :nHar] = hAmp
            hPhaseList[iHop, :nHar] = hPhase
        return hFreqList, hAmpList, hPhaseList