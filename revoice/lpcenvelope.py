import numpy as np
import scipy.signal as sp
import scipy.interpolate as ipl

from .common import *
from . import lpc

class Analyzer:
    def __init__(self, sr, **kwargs):
        defaultOrder = int(np.ceil(sr / 12000 * 13))
        if(defaultOrder % 2 == 0):
            defaultOrder += 1
        
        self.samprate = float(sr)
        self.hopSize = kwargs.get("hopSize", self.samprate * 0.0025)
        self.fftSize = kwargs.get("fftSize", roundUpToPowerOf2(self.samprate * 0.05))
        self.order = kwargs.get("order", defaultOrder)
        self.window = getWindow(kwargs.get("window", "blackman"))
        self.lpcAnalysisMethod = kwargs.get("lpcAnalysisMethod", "burg")
        self.preemphasisFreq = kwargs.get("lpcAnalysisMethod", 50.0)

    def __call__(self, x, f0List):
        nBin = self.fftSize // 2 + 1
        (nHop,) = f0List.shape
        (nX,) = x.shape
        assert getNFrame(nX, self.hopSize) == nHop

        if(self.preemphasisFreq):
            x = applyPreEmphasisFilter(x, self.preemphasisFreq, self.samprate)

        lpcProc = lpc.Analyzer(self.samprate, lpcAnalysisMethod = self.lpcAnalysisMethod)
        coeffList, xmsList = lpcProc(x, f0List, self.order)

        lpcSpectrum = np.zeros((nHop, nBin))
        for iHop in range(nHop):
            lpcSpectrum[iHop] = lpc.calcMagnitudeFromLPC(coeffList[iHop], xmsList[iHop], self.fftSize, self.samprate, deEmphasisFreq = self.preemphasisFreq)
        
        return np.log(np.clip(lpcSpectrum, eps, np.inf))
