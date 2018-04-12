import numpy as np
import scipy.signal as sp

from .common import *

class Analyzer:
    def __init__(self, sr, **kwargs):
        self.samprate = float(sr)
        self.hopSize = kwargs.get("hopSize", self.samprate * 0.0025)
        self.fftSize = kwargs.get("fftSize", roundUpToPowerOf2(self.samprate * 0.05))

        self.filterKernelSize = kwargs.get("filterKernelSize", 65)
        self.filterCutoff = kwargs.get("filterCutoff", 0.03)
        self.filterTransExp = kwargs.get("filterTransExp", 8)
        self.windowLengthFac = kwargs.get("windowLengthFac", 1.0)
        self.fixedF0 = kwargs.get("fixedF0", 220.0)

    def __call__(self, x, f0List):
        # constant
        nX = len(x)
        nHop = getNFrame(nX, self.hopSize)
        nBin = self.fftSize // 2 + 1
        halfKernelSize = self.filterKernelSize // 2

        # check input
        assert (nHop,) == f0List.shape
        assert self.fftSize % 2 == 0

        # do calculate
        out = np.zeros((nHop, nBin))
        kernel = sp.firwin(self.filterKernelSize, self.filterCutoff, window = "hanning", pass_zero = True)
        trans = (np.arange(halfKernelSize) / (halfKernelSize - 1)) ** self.filterTransExp
        revTrans = trans[::-1]
        for iHop, f0 in enumerate(f0List):
            if(f0 <= 0.0):
                f0 = self.fixedF0
            iCenter = int(round(iHop * self.hopSize))
            offsetRadius = int(np.ceil(self.samprate / (2 * f0)))
            stdev = self.samprate / (3 * f0)

            windowSize = int(2 * self.samprate / f0 * self.windowLengthFac)
            if(windowSize % 2 != 0):
                windowSize += 1
            window = sp.gaussian(windowSize, stdev)
            window *= 2 / np.sum(window)
            maxSpec = np.full(nBin, -np.inf)
            minBin = np.full(nBin, np.inf)
            integratedSpec = np.zeros(nBin)
            for offset in range(-offsetRadius, offsetRadius):
                frame = getFrame(x, iCenter + offset, windowSize) * window
                spec = np.abs(np.fft.rfft(frame, n = self.fftSize))
                need = maxSpec < spec
                maxSpec[need] = spec[need]
                need = spec < minBin
                minBin[need] = spec[need]
                integratedSpec += spec
            integratedSpec /= 2 * offsetRadius
            integratedEnergy = np.sum(integratedSpec ** 2)
            if(integratedEnergy < 1e-16):
                out[iHop] = 1e-6
                continue
            integratedSpec = np.log(np.clip(integratedSpec, 1e-6, np.inf))
            smoothedSpec = np.convolve(integratedSpec, kernel)[halfKernelSize:-halfKernelSize]
            smoothedSpec[:halfKernelSize] = integratedSpec[:halfKernelSize] + (smoothedSpec[:halfKernelSize] - integratedSpec[:halfKernelSize]) * trans
            smoothedSpec[-halfKernelSize:] = integratedSpec[-halfKernelSize:] + (smoothedSpec[-halfKernelSize:] - integratedSpec[-halfKernelSize:]) * revTrans
            linearSmoothedSpec = np.exp(smoothedSpec)
            smoothedSpec = np.log(linearSmoothedSpec * np.sqrt(integratedEnergy / np.sum(linearSmoothedSpec ** 2)))
            out[iHop] = smoothedSpec

        return out
