from .common import *

class Analyzer:
    def __init__(self, sr, **kwargs):
        self.samprate = float(sr)
        self.hopSize = kwargs.get("hopSize", self.samprate * 0.0025)
        self.fftSize = kwargs.get("fftSize", roundUpToPowerOf2(self.samprate * 0.05))
        self.peakSearchRange = kwargs.get("peakSearchRange", 0.3)
        self.window = kwargs.get("window", "blackman")
        self.windowLengthFac = kwargs.get("windowLengthFac", 1.0)
        self.removeDC = kwargs.get("removeDC", True)

    def __call__(self, x, f0List):
        # constant
        (nX,) = x.shape
        nHop = getNFrame(nX, self.hopSize)
        nBin = self.fftSize // 2 + 1
        windowFunc, B, _ = getWindow(self.window)
        B *= self.windowLengthFac

        # check input
        assert (nHop,) == f0List.shape
        assert self.hopSize > 0
        assert self.fftSize > 0 and roundUpToPowerOf2(self.fftSize) == self.fftSize

        # do calculate
        fSigList = np.zeros((nHop, nBin), dtype = np.complex128)
        for iHop, f0 in enumerate(f0List):
            iCenter = int(round(iHop * self.hopSize))
            if(f0 > 0.0):
                windowSize = int(min(self.fftSize, self.samprate / f0 * B * 2.0))
            else:
                windowSize = int(self.hopSize * 2)
            if(windowSize % 2 != 0):
                windowSize += 1
            
            halfWindowSize = windowSize // 2
            window = windowFunc(windowSize)
            windowNormFac = 2.0 / np.sum(window)
            frame = getFrame(x, iCenter, windowSize)
            if(self.removeDC):
                frame = removeDCSimple(frame)
            frame *= window

            tSig = np.zeros(self.fftSize, dtype = np.float64)
            tSig[:halfWindowSize] = frame[halfWindowSize:]
            tSig[-halfWindowSize:] = frame[:halfWindowSize]
            fSig = np.fft.rfft(tSig)
            fSig *= windowNormFac
            fSigList[iHop] = fSig
        
        return fSigList
