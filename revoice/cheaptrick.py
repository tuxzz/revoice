from .common import *
from . import adaptivestft

def calcCheapTrick(x, f0, sr, order):
    assert order % 2 == 1 # for mavg filter
    nX = len(x)
    nFFT = (nX - 1) * 2

    x = x.copy()
    iF0 = int(round(f0 / sr * nFFT))
    x[:iF0 // 2] = x[iF0 - iF0 // 2:iF0][::-1]
    smoothed = np.log(np.clip(applyMovingAverageFilter(x, order), eps, np.inf))

    c = np.fft.irfft(smoothed)
    a = np.arange(1, nX) * (f0 / sr * np.pi)
    c[1:nX] *= np.sin(a) / a * (1.18 - 2.0 * 0.09 * np.cos(np.arange(1, nX) * (2.0 * np.pi * f0 / sr)))
    c[nX:] = c[1:nX - 1][::-1]
    smoothed = np.fft.rfft(c).real

    return smoothed.astype(np.float32)

class Analyzer:
    def __init__(self, sr, **kwargs):
        self.samprate = float(sr)
        self.orderFac = kwargs.get("orderFac", 1.0)
        self.fixedF0 = kwargs.get("fixedF0", 220.0)

    def __call__(self, magnList, f0List):
        # early check input
        assert magnList.ndim == 2
        assert (magnList.shape[0],) == f0List.shape

        # constant
        nHop, nBin = magnList.shape
        fftSize = (nBin - 1) * 2

        # check input
        assert nHop == f0List.shape[0]
        out = np.zeros(magnList.shape, dtype=np.float32)
        for iHop, f0 in enumerate(f0List):
            if(f0 <= 0.0):
                f0 = self.fixedF0
            order = int(f0 / self.samprate * fftSize / 3.0 * self.orderFac)
            if(order % 2 == 0):
                order += 1
            out[iHop] = calcCheapTrick(magnList[iHop], f0, self.samprate, order)

        return out
