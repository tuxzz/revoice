from .common import *

def _refineF0STFTCore(x, inOutF0List, hopSize, fftSize, B, sr, peakSearchRange):
    f0List = inOutF0List
    for iHop, f0 in enumerate(f0List):
        if f0 <= 0.0:
            continue
        iCenter = int(round(iHop * hopSize))
        # fft
        windowSize = int(min(fftSize, sr / f0 * B * 2.0))
        if(windowSize % 2 != 0):
            windowSize += 1
        halfWindowSize = windowSize // 2

        window = blackman(windowSize)
        frame = getFrame(x, iCenter, windowSize)
        frame = removeDCSimple(frame)
        frame *= window

        tSig = np.zeros(fftSize, dtype = np.float64)
        tSig[:halfWindowSize] = frame[halfWindowSize:]
        tSig[-halfWindowSize:] = frame[:halfWindowSize]
        fSig = np.fft.rfft(tSig)
        magn = np.abs(fSig)

        # find peak
        lowerIdx = max(1, int(np.floor(f0 * fftSize / sr * (1.0 - peakSearchRange))))
        upperIdx = min(fftSize // 2, int(np.floor(f0 * fftSize / sr * (1.0 + peakSearchRange))))
        peakIdx = np.argmax(magn[lowerIdx:upperIdx]) + lowerIdx

        # optimize peak
        costFunction = lambda f:-np.log(np.abs(calcSpectrumAtFreq(frame, np.array((f,), dtype=np.float32), sr)[0]))
        peakFreq = fmin_scalar(costFunction, ((peakIdx - 1.0) / fftSize * sr, (peakIdx + 1.0) / fftSize * sr), 32)

        peakPhase = np.angle(calcSpectrumAtFreq(frame, np.array((peakFreq,), dtype=np.float32), sr)[0])
        peakPhase -= np.floor(peakPhase / 2.0 / np.pi) * 2.0 * np.pi
        # peak delta phase
        frame = getFrame(x, iCenter - 1, windowSize)
        frame = removeDCSimple(frame)
        frame *= window

        prevPeakPhase = np.angle(calcSpectrumAtFreq(frame, np.array((peakFreq,), dtype=np.float32), sr)[0])
        prevPeakPhase -= np.floor(prevPeakPhase / 2.0 / np.pi) * 2.0 * np.pi
        if peakPhase < prevPeakPhase:
            peakPhase += 2 * np.pi
        assert peakPhase >= prevPeakPhase
        refinedF0 = (peakPhase - prevPeakPhase) / 2.0 / np.pi * sr
        if np.abs(refinedF0 - peakFreq) > sr / fftSize * 1.5:
            print("Reject[%d] Δ = %f / %f" % (iHop, refinedF0 - peakFreq, refinedF0 - f0))
            f0List[iHop] = peakFreq
        else:
            f0List[iHop] = refinedF0

class Processor:
    def __init__(self, sr, **kwargs):
        self.samprate = sr
        self.hopSize = kwargs.get("hopSize", sr * 0.0025)

        self.fftSize = kwargs.get("fftSize", roundUpToPowerOf2(sr * 0.05))
        self.peakSearchRange = kwargs.get("peakSearchRange", 0.3)
        self.windowLengthFac = kwargs.get("windowLengthFac", 1.0)
        self.useAccelerator = kwargs.get("useAccelerator", True)

        assert self.fftSize % 2 == 0
        assert isinstance(self.fftSize, int)
    
    def __call__(self, x, f0List):
        (nX,) = x.shape
        nHop = getNFrame(nX, self.hopSize)
        _, B, _ = getWindow("blackman")
        B *= self.windowLengthFac

        assert (nHop,) == f0List.shape
        assert self.fftSize == roundUpToPowerOf2(self.fftSize)

        f0List = f0List.copy().astype(np.float32)
        if self.useAccelerator:
            try:
                from . import accelerator
                accelerator.refineF0STFTCore(x.astype(np.float32), f0List, self.hopSize, self.fftSize, B, self.samprate, self.peakSearchRange)
            except Exception as e:
                print("[ERROR] Failed to call accelerator, fallback: %s" % (str(e),))
                _refineF0STFTCore(x, f0List, self.hopSize, self.fftSize, B, self.samprate, self.peakSearchRange)
        else:
            _refineF0STFTCore(x, f0List, self.hopSize, self.fftSize, B, self.samprate, self.peakSearchRange)
        return f0List