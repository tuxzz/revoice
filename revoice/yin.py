from .common import *
import pylab as pl

@nb.jit(nb.float32[:](nb.float32[:]), fastmath=True, cache=True)
def difference(x):
    (frameSize,) = x.shape
    assert frameSize % 2 == 0
    
    paddedSize = roundUpToPowerOf2(frameSize)
    outSize = frameSize // 2

    # calculate the power term, see yin paper eq. (7)
    powerTerms = np.zeros(outSize, dtype=np.float32)
    powerTerms[0] = np.sum(x[:outSize] ** 2)

    for i in range(1, outSize):
        powerTerms[i] = powerTerms[i - 1] - x[i - 1] * x[i - 1] + x[i + outSize] * x [i + outSize]

    # Yin-style ACF via FFT
    # 1. data
    transformedAudio = np.fft.rfft(x, n=paddedSize)

    # 2. half of the data, disguised as a convolution kernel
    kernel = x[:outSize][::-1]
    transformedKernel = np.fft.rfft(kernel, n=paddedSize)

    # 3. convolution
    yinStyleACF = transformedAudio * transformedKernel
    correlation = np.fft.irfft(yinStyleACF)

    # calculate difference function according to (7) in the Yin paper
    out = powerTerms[0] + powerTerms - 2 * correlation[outSize - 1:frameSize - 1]
    return out.astype(np.float32)

@nb.jit((nb.float32[:],), fastmath=True, nopython=True, cache=True)
def cumulativeDifference(out):
    (nOut,) = out.shape

    out[0] = 1.0
    sumValue = 0.0

    for i in range(1, nOut):
        sumValue += out[i]
        if sumValue == 0.0:
            out[i] = 1.0
        else:
            out[i] *= i / sumValue

@nb.jit(fastmath=True, nopython=True, cache=True)
def findValleys(x, minFreq, maxFreq, sr, threshold = 0.5, step = 0.01, limit = 64):
    ret = []
    begin = max(1, int(sr / maxFreq))
    end = min(len(x) - 1, int(np.ceil(sr / minFreq)))
    for i in range(begin, end):
        prev = x[i - 1]
        curr = x[i]
        next = x[i + 1]
        if prev > curr and next > curr and curr < threshold:
            threshold = curr - step
            ret.append(i)
            if len(ret) >= limit:
                break
    return ret

def applyPrefilter(x, maxFreq, sr):
    filterOrder = int(2048 * sr / 44100.0)
    if(filterOrder % 2 == 0):
        filterOrder += 1
    f = sp.firwin(filterOrder, max(maxFreq + 500.0, maxFreq * 3.0), window="blackman", fs=sr)
    halfFilterOrder = filterOrder // 2
    x = sp.fftconvolve(x, f)[halfFilterOrder:-halfFilterOrder]
    return x.astype(np.float32)

class Analyzer:
    def __init__(self, sr, **kwargs):
        self.samprate = float(sr)
        self.hopSize = kwargs.get("hopSize", self.samprate * 0.0025)

        self.minFreq = kwargs.get("minFreq", 80.0)
        self.maxFreq = kwargs.get("maxFreq", 1000.0)
        self.windowSize = kwargs.get("windowSize", max(roundUpToPowerOf2(self.samprate / self.minFreq * 2), self.hopSize * 4))
        self.prefilter = kwargs.get("prefilter", True)

        self.valleyThreshold = kwargs.get("valleyThreshold", 0.5)
        self.valleyStep = kwargs.get("valleyStep", 0.01)
        self.removeDC = kwargs.get("removeDC", True)

    def __call__(self, x):
        (nX,) = x.shape
        nHop = getNFrame(nX, self.hopSize)

        if self.prefilter:
            x = applyPrefilter(x, self.maxFreq, self.samprate)

        out = np.zeros(nHop, dtype=np.float32)
        for iHop in range(nHop):
            iCenter = int(round(iHop * self.hopSize))
            frame = getFrame(x, iCenter, self.windowSize)
            if self.removeDC:
                frame = removeDCSimple(frame)
            buff = difference(frame)
            cumulativeDifference(buff)
            valleyIndexList = findValleys(buff, self.minFreq, self.maxFreq, self.samprate, threshold = self.valleyThreshold, step = self.valleyStep)
            out[iHop] = self.samprate / parabolicInterpolate(buff, valleyIndexList[-1], False)[0] if(valleyIndexList) else 0.0

        return out
