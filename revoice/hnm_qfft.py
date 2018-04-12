from .common import *
from . import adaptivestft

class Analyzer:
    def __init__(self, hopSize, mvf, sr, **kwargs):
        self.samprate = float(sr)
        self.hopSize = hopSize
        self.window = kwargs.get("window", "blackman")
        self.fftSize = kwargs.get("fftSize", roundUpToPowerOf2(self.samprate * 0.05))
        self.windowLengthFac = kwargs.get("windowLengthFac", 1.0)
        self.strictHarmonic = True

        self.mvf = float(mvf)
        self.maxHarmonicOffset = kwargs.get("maxHarmonicOffset", 0.25)

        assert(self.mvf <= self.samprate / 2)

    def __call__(self, x, f0List, maxHar):
        # constant
        (nX,) = x.shape
        nHop = getNFrame(nX, self.hopSize)

        # check input
        assert (nHop,) == f0List.shape

        # do stft
        stftProc = adaptivestft.Analyzer(self.samprate, window = self.window, hopSize = self.hopSize, fftSize = self.fftSize, windowLengthFac = self.windowLengthFac)
        fSigList = stftProc(x, f0List)
        fSigList.T[0] = 0.0
        magnList = np.log(np.clip(np.abs(fSigList), eps, np.inf))
        phaseList = np.angle(fSigList)
        del fSigList

        # find quasi-harmonic
        hFreqList = np.zeros((nHop, maxHar), dtype = np.float64)
        hAmpList = np.zeros((nHop, maxHar), dtype = np.float64)
        hPhaseList = np.zeros((nHop, maxHar), dtype = np.float64)
        for iHop, f0 in enumerate(f0List):
            if(f0 <= 0.0):
                continue
            hFreqList[iHop], hAmpList[iHop], hPhaseList[iHop] = self._findHarmonic(f0, magnList[iHop], phaseList[iHop], maxHar)
        hAmpList = np.exp(hAmpList)

        return hFreqList, hAmpList, hPhaseList

    def _findHarmonic(self, f0, magn, phase, maxHar):
        (nBin,) = magn.shape

        assert (nBin - 1) * 2 == self.fftSize
        assert magn.shape == phase.shape
        assert f0 > 0.0

        hFreq = np.zeros(maxHar, dtype = np.float64)
        hAmp = np.full(maxHar, -np.inf, dtype = np.float64)
        hPhase = np.zeros(maxHar, dtype = np.float64)

        nHar = min(maxHar, int(self.mvf / f0))
        offset = f0 * self.maxHarmonicOffset
        for iHar in range(1, nHar + 1):
            freq = iHar * f0
            if(freq >= self.mvf):
                break
            lowerIdx = max(0, int(np.floor((freq - offset) / self.samprate * self.fftSize)))
            upperIdx = min(nBin - 1, int(np.ceil((freq + offset) / self.samprate * self.fftSize)))
            peakBin = findPeak(magn, lowerIdx, upperIdx)
            ipledPeakBin, ipledPeakAmp = parabolicInterpolate(magn, peakBin)
            hFreq[iHar - 1] = freq if(self.strictHarmonic) else ipledPeakBin * self.samprate / self.fftSize
            hAmp[iHar - 1] = ipledPeakAmp
            hPhase[iHar - 1] = lerp(phase[peakBin], phase[peakBin + 1], ipledPeakBin - np.floor(ipledPeakBin))

        return hFreq, hAmp, hPhase
