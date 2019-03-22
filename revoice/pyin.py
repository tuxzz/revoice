from .common import *
from . import yin

@nb.jit(nb.float32[:](nb.float32, nb.float32, nb.float32, nb.float32, nb.int64))
def normalized_pdf(a, b, begin, end, number):
    x = np.arange(0, number, dtype=np.float32) * ((end - begin) / number)
    v = np.power(x, a - 1.0) * np.power(1.0 - x, b - 1.0)
    for i in reversed(range(number - 1)):
        if(v[i] < v[i + 1]):
            v[i] = v[i + 1]
    return v / np.sum(v)

@nb.jit(nb.float32[:](nb.float32[:]))
def extractF0(obsProbList):
    nHop = len(obsProbList)

    out = np.zeros(nHop, dtype=np.float32)
    for iHop, (freqProb) in enumerate(obsProbList):
        if len(freqProb) > 0:
            out[iHop] = freqProb.T[0][np.argmax(freqProb.T[1])]

    return out

@nb.jit(fastmath=True, nopython=True, cache=True)
def _valleyToFreqProb(valleyIndexList, buff, outFreqProb, pdf, bias, probThreshold, weightPrior, sr):
    (nValley,) = valleyIndexList.shape
    (pdfSize,) = pdf.shape
    freqProb = outFreqProb
    probTotal = 0.0
    weightedProbTotal = 0.0
    for iValley, valley in enumerate(valleyIndexList):
        ipledIdx, ipledVal = parabolicInterpolate(buff, valley, False)
        freq = sr / ipledIdx
        v0 = 1 if(iValley == 0) else min(1.0, max(0.0, buff[valleyIndexList[iValley - 1]]) + eps)
        v1 = 0 if(iValley == nValley - 1) else min(1.0, max(0.0, buff[valleyIndexList[iValley + 1]]) + eps)
        prob = 0.0
        for i in range(int(v1 * pdfSize), int(v0 * pdfSize)):
            prob += pdf[i] * (1.0 if(ipledVal < i / pdfSize) else 0.01)
        prob = min(prob, 0.99)
        prob *= bias
        probTotal += prob
        if(ipledVal < probThreshold):
            prob *= weightPrior
        weightedProbTotal += prob
        freqProb[iValley] = freq, prob
    return (probTotal, weightedProbTotal)

def _pyinCore(x: np.ndarray, pdf: np.ndarray, bias: float, probThreshold: float, weightPrior: float, hopSize: float, valleyThreshold: float, valleyStep: float, minFreq: float, maxFreq: float, minWindowSize: float, windowLengthFac: float, sr: float, maxIter: int, removeDC: bool, out: np.ndarray) -> np.ndarray:
    (nHop, maxFreqProbCount, _) = out.shape
    
    for iHop in range(nHop):
        iCenter = int(round(iHop * hopSize))
        windowSize = int(max(sr / minFreq * 4, minWindowSize) * windowLengthFac)
        if(windowSize % 2 == 1):
            windowSize += 1
        for _ in range(maxIter):
            frame = getFrame(x, iCenter, windowSize)
            if removeDC:
                frame = removeDCSimple(frame)
            buff = yin.difference(frame)
            yin.cumulativeDifference(buff)
            valleyIndexList = np.asarray(yin.findValleys(buff, minFreq, maxFreq, sr, threshold = valleyThreshold, step = valleyStep, limit = maxFreqProbCount), dtype = np.int)
            nValley = len(valleyIndexList)
            if nValley > 0:
                possibleFreq = min(maxFreq, max(sr / valleyIndexList[-1] - 20.0, minFreq))
                newWindowSize = int(max(sr / possibleFreq * 4, minWindowSize) * windowLengthFac)
                if newWindowSize % 2 == 1:
                    newWindowSize += 1
                if newWindowSize == windowSize:
                    break
                windowSize = newWindowSize
        freqProb = np.zeros((nValley, 2), dtype=np.float32)
        (probTotal, weightedProbTotal) = _valleyToFreqProb(valleyIndexList, buff, freqProb, pdf, bias, probThreshold, weightPrior, sr)

        # renormalize
        if nValley > 0 and weightedProbTotal != 0.0:
            freqProb.T[1] *= probTotal / weightedProbTotal
        
        out[iHop,:nValley] = freqProb

class Analyzer:
    def __init__(self, sr, **kwargs):
        self.samprate = float(sr)
        self.hopSize = kwargs.get("hopSize", self.samprate * 0.0025)

        self.minFreq = kwargs.get("minFreq", 80.0)
        self.maxFreq = kwargs.get("maxFreq", 1000.0)
        self.maxIter = 4
        self.prefilter = kwargs.get("prefilter", True)

        self.valleyThreshold = kwargs.get("valleyThreshold", 1.0)
        self.valleyStep = kwargs.get("valleyStep", 0.01)

        self.probThreshold = kwargs.get("probThreshold", 0.02)
        self.weightPrior = kwargs.get("weightPrior", 5.0)
        self.bias = kwargs.get("bias", 1.0)

        self.pdf = kwargs.get("pdf", normalized_pdf(1.7, 6.8, 0.0, 1.0, 128))
        self.removeDC = kwargs.get("removeDC", True)
        self.windowLengthFac = kwargs.get("windowLengthFac", 1.0)
        self.minWindowSize = kwargs.get("minWindowSize", self.hopSize * 3)
        self.maxFreqProbCount = kwargs.get("maxFreqProbCount", 16)
        self.useAccelerator = kwargs.get("useAccelerator", True)

        self.pdf = np.require(self.pdf, dtype=np.float32, requirements="C")

    def __call__(self, x):
        (nX,) = x.shape
        nHop = getNFrame(nX, self.hopSize)

        if self.prefilter:
            x = yin.applyPrefilter(x, self.maxFreq, self.samprate)
        
        out = np.zeros((nHop, self.maxFreqProbCount, 2), dtype = np.float32)
        if self.useAccelerator:
            try:
                from . import accelerator
                accelerator.pyinCore(x, self.pdf, self.bias, self.probThreshold, self.weightPrior, self.hopSize, self.valleyThreshold, self.valleyStep, self.minFreq, self.maxFreq, self.minWindowSize, self.windowLengthFac, self.samprate, self.maxIter, self.removeDC, out)
            except Exception as e:
                print("[ERROR] Failed to call accelerator, fallback: %s" % (str(e),))
                _pyinCore(x, self.pdf, self.bias, self.probThreshold, self.weightPrior, self.hopSize, self.valleyThreshold, self.valleyStep, self.minFreq, self.maxFreq, self.minWindowSize, self.windowLengthFac, self.samprate, self.maxIter, self.removeDC, out)
        else:
            _pyinCore(x, self.pdf, self.bias, self.probThreshold, self.weightPrior, self.hopSize, self.valleyThreshold, self.valleyStep, self.minFreq, self.maxFreq, self.minWindowSize, self.windowLengthFac, self.samprate, self.maxIter, self.removeDC, out)
        return out
