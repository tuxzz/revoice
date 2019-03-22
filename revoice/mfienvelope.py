import numpy as np
import scipy.signal as sp

from .common import *

def _mfiCore(x, f0List, kernel, trans, fixedF0, hopSize, sr, windowLengthFac, fftSize, out):
    nBin = fftSize // 2 + 1
    (halfKernelSize,) = trans.shape
    revTrans = trans[::-1]
    for iHop, f0 in enumerate(f0List):
        if f0 <= 0.0:
            f0 = fixedF0
        # generate window
        iCenter = int(round(iHop * hopSize))
        offsetRadius = int(np.ceil(sr / (2 * f0)))
        stdev = sr / (3 * f0)
        windowSize = min(int(2 * sr / f0 * windowLengthFac), fftSize)
        if(windowSize % 2 != 0):
            windowSize += 1
        window = gaussian(windowSize, stdev)
        window *= 2 / np.sum(window)

        # calc average(integrated) magn
        integratedMagn = np.zeros(nBin)
        for offset in range(-offsetRadius, offsetRadius):
            frame = removeDCSimple(getFrame(x, iCenter + offset, windowSize)) * window
            integratedMagn += np.abs(np.fft.rfft(frame, n = fftSize))
        integratedMagn /= 2 * offsetRadius
        integratedEnergy = np.sum(integratedMagn ** 2)
        if(integratedEnergy < 1e-16):
            out[iHop] = 1e-6
            continue
        
        # filter average magn on log domain
        integratedMagn = np.log(np.clip(integratedMagn, 1e-6, np.inf))
        smoothedMagn = np.convolve(integratedMagn, kernel)[halfKernelSize:-halfKernelSize]
        # make bounds better
        smoothedMagn[:halfKernelSize] = integratedMagn[:halfKernelSize] + (smoothedMagn[:halfKernelSize] - integratedMagn[:halfKernelSize]) * trans
        smoothedMagn[-halfKernelSize:] = integratedMagn[-halfKernelSize:] + (smoothedMagn[-halfKernelSize:] - integratedMagn[-halfKernelSize:]) * revTrans
        # normalize filtered magn and output
        smoothedMagn += np.log(np.sqrt(integratedEnergy / np.sum(np.exp(smoothedMagn) ** 2)))
        out[iHop] = smoothedMagn
    return out

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
        self.useAccelerator = kwargs.get("useAccelerator", True)

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
        kernel = sp.firwin(self.filterKernelSize, self.filterCutoff, window="hanning", pass_zero=True).astype(np.float32)
        trans = (np.arange(halfKernelSize, dtype=np.float32) / (halfKernelSize - 1)) ** self.filterTransExp
        out = np.zeros((nHop, nBin), dtype=np.float32)
        if self.useAccelerator:
            try:
                from . import accelerator
                accelerator.mfiCore(x, f0List, kernel, trans, self.fixedF0, self.hopSize, self.samprate, self.windowLengthFac, self.fftSize, out)
            except Exception as e:
                print("[ERROR] Failed to call accelerator, fallback: %s" % (str(e),))
                _mfiCore(x, f0List, kernel, trans, self.fixedF0, self.hopSize, self.samprate, self.windowLengthFac, self.fftSize, out)
        else:
            _mfiCore(x, f0List, kernel, trans, self.fixedF0, self.hopSize, self.samprate, self.windowLengthFac, self.fftSize, out)
        return out
