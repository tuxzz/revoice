from .common import *
from . import instantfrequency

class Processor:
    def __init__(self, sr, **kwargs):
        self.samprate = sr
        self.hopSize = kwargs.get("hopSize", self.samprate * 0.0025)
        self.maxAdjustmentRatio = kwargs.get("maxAdjustmentRatio", 0.25)
        self.removeDC = kwargs.get("removeDC", True)

        self.fftSize = kwargs.get("fftSize", roundUpToPowerOf2(self.samprate * 0.05))
        self.peakSearchRange = kwargs.get("peakSearchRange", 0.3)
        self.window = kwargs.get("window", "blackman")
        self.windowLengthFac = kwargs.get("windowLengthFac", 1.0)
        self.gridSearchPointList = kwargs.get("gridSearchPointList", np.linspace(-1.0, 1.0, 32))

        assert self.fftSize % 2 == 0
    
    def __call__(self, x, f0List):
        (nX,) = x.shape
        nHop = getNFrame(nX, self.hopSize)
        windowFunc, B, windowMean = getWindow(self.window)
        B *= self.windowLengthFac

        assert (nHop,) == f0List.shape
        assert self.fftSize == roundUpToPowerOf2(self.fftSize)

        f0List = f0List.copy()
        for iHop, f0 in enumerate(f0List):
            if(f0 < 0.0):
                continue
            iCenter = int(round(iHop * self.hopSize))
            # fft
            windowSize = int(min(self.fftSize, self.samprate / f0 * B * 2.0))
            if(windowSize % 2 != 0):
                windowSize += 1
            halfWindowSize = windowSize // 2

            window = windowFunc(windowSize)
            windowNormFac = 2.0 / (windowMean * windowSize)
            frame = getFrame(x, iCenter, windowSize)
            frame *= window * windowNormFac
            if(self.removeDC):
                frame = removeDCSimple(frame)

            tSig = np.zeros(self.fftSize, dtype = np.float64)
            tSig[:halfWindowSize] = frame[halfWindowSize:]
            tSig[-halfWindowSize:] = frame[:halfWindowSize]
            fSig = np.fft.rfft(tSig)
            magn = np.abs(fSig)
            phase = np.unwrap(np.angle(fSig))

            # find peak
            lowerIdx = max(0, int(np.floor(f0 * self.fftSize / self.samprate * (1.0 - self.peakSearchRange))))
            upperIdx = min(self.fftSize // 2, int(np.floor(f0 * self.fftSize / self.samprate * (1.0 + self.peakSearchRange))))
            peakIdx = np.argmax(magn[lowerIdx:upperIdx]) + lowerIdx

            #del tSig, fSig, magn, phase

            # optimize peak
            costFunction = lambda f:-np.log(np.abs(calcSpectrumAtFreq(frame, np.array((f,)), self.samprate)[0]))
            gridSearchPointList = (peakIdx + self.gridSearchPointList) / self.fftSize * self.samprate
            peakFreq = minimizeScalar(costFunction, gridSearchPointList)

            peakPhase = np.angle(calcSpectrumAtFreq(frame, np.array((peakFreq,)), self.samprate)[0])
            peakPhase -= np.floor(peakPhase / 2.0 / np.pi) * 2.0 * np.pi
            # delta peak phase
            frame = getFrame(x, iCenter - 1, windowSize)
            frame *= window
            if(self.removeDC):
                frame = removeDCSimple(frame)

            prevPeakPhase = np.angle(calcSpectrumAtFreq(frame, np.array((peakFreq,)), self.samprate)[0])
            prevPeakPhase -= np.floor(prevPeakPhase / 2.0 / np.pi) * 2.0 * np.pi
            if(peakPhase < prevPeakPhase):
                peakPhase += 2 * np.pi
            assert peakPhase >= prevPeakPhase
            refinedF0 = (peakPhase - prevPeakPhase) / 2.0 / np.pi * self.samprate
            if(np.abs(refinedF0 - f0) / f0 > self.maxAdjustmentRatio):# or np.abs(refinedF0 - f0) > 1 / self.fftSize * self.samprate):
                f0List[iHop] = f0
            else:
                f0List[iHop] = refinedF0
        return f0List