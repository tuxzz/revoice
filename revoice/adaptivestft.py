from .common import *

def calcScaledSpectrumFilterBank(funcUnitToFreq, nFilter, firstFilter, filterDistance, fftSize, sr):
    assert fftSize % 2 == 0

    nBin = fftSize // 2 + 1
    freqRange = np.fft.rfftfreq(fftSize, 1.0 / sr)

    filterBank = np.zeros((nFilter, nBin), dtype=np.float32)
    for iFilter in range(nFilter):
        peakUnit = firstFilter + iFilter * filterDistance
        peakFreq = funcUnitToFreq(peakUnit)
        leftFreq = funcUnitToFreq(peakUnit - filterDistance)
        rightFreq = funcUnitToFreq(peakUnit + filterDistance)
        iplX = (leftFreq, peakFreq, rightFreq)
        iplY = (0.0, 1.0, 0.0)
        filterBank[iFilter] = ipl.interp1d(iplX, iplY, kind="linear", bounds_error=False, fill_value=0.0)(freqRange)
        bankSum = np.sum(filterBank[iFilter])
        if(bankSum > 0.0):
            filterBank[iFilter] /= bankSum
    return filterBank

def calcMelScaledSpectrumFilterBank(nFilter, firstFilter, filterDistance, fftSize, sr):
    return calcScaledSpectrumFilterBank(melToFreq, nFilter, firstFilter, filterDistance, fftSize, sr)

def calcBarkScaledSpectrumFilterBank(nFilter, firstFilter, filterDistance, fftSize, sr):
    return calcScaledSpectrumFilterBank(barkToFreq, nFilter, firstFilter, filterDistance, fftSize, sr)

def applyFilterBank(magn, filterBank):
    (nFilter, nFilterBin) = filterBank.shape
    assert nFilter > 0, "filter count must be greater than 0"
    if magn.ndim == 1:
        (nHop, nBin) = (1, *magn.shape)
    elif magn.ndim:
        (nHop, nBin) = magn.shape
    assert nBin == nFilterBin, "shape mismatch between specgram and filterBank"

    out = np.sum(magn.reshape(nHop, 1, nBin) * filterBank.reshape(1, nFilter, nBin), axis = 2)
    if magn.ndim == 1:
        return out.reshape(out.shape[1])
    else:
        return out

def calcLinearMagnFromScaled(scaledMagn, funcUnitToFreq, nFilter, firstFilter, filterDistance, fftSize, sr):
    assert fftSize % 2 == 0
    assert scaledMagn.shape[-1] == nFilter, "bad shape of scaledMagn"

    freqRange = np.fft.rfftfreq(fftSize, 1.0 / sr)

    linearX = np.zeros(nFilter, dtype=np.float32)
    for iFilter in range(nFilter):
        linearX[iFilter] = funcUnitToFreq(iFilter * filterDistance + firstFilter)

    return ipl.interp1d(linearX, scaledMagn, kind="linear", axis=-1, bounds_error=False, fill_value=0.0)(freqRange).astype(np.float32)

def calcLinearMagnFromMel(scaledMagn, nFilter, firstFilter, filterDistance, fftSize, sr):
    return calcLinearMagnFromScaled(scaledMagn, melToFreq, nFilter, firstFilter, filterDistance, fftSize, sr)

def calcLinearMagnFromBark(scaledMagn, nFilter, firstFilter, filterDistance, fftSize, sr):
    return calcLinearMagnFromScaled(scaledMagn, barkToFreq, nFilter, firstFilter, filterDistance, fftSize, sr)

class Analyzer:
    def __init__(self, sr, **kwargs):
        self.samprate = float(sr)
        self.hopSize = kwargs.get("hopSize", self.samprate * 0.0025)
        self.fftSize = kwargs.get("fftSize", roundUpToPowerOf2(self.samprate * 0.05))
        self.peakSearchRange = kwargs.get("peakSearchRange", 0.3)
        self.window = kwargs.get("window", "blackman")
        self.windowLengthFac = kwargs.get("windowLengthFac", 1.0)
        self.removeDC = kwargs.get("removeDC", True)

        assert isinstance(self.fftSize, int)

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
        fSigList = np.zeros((nHop, nBin), dtype = np.complex64)
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

            tSig = np.zeros(self.fftSize, dtype = np.float32)
            tSig[:halfWindowSize] = frame[halfWindowSize:]
            tSig[-halfWindowSize:] = frame[:halfWindowSize]
            fSig = np.fft.rfft(tSig)
            fSig *= windowNormFac
            fSigList[iHop] = fSig
        
        return fSigList
