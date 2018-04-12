from .common import *
from . import yin

def normalized_pdf(a, b, begin, end, number):
    x = np.arange(0, number, dtype = np.float64) * ((end - begin) / number)
    v = np.power(x, a - 1.0) * np.power(1.0 - x, b - 1.0)
    for i in reversed(range(number - 1)):
        if(v[i] < v[i + 1]):
            v[i] = v[i + 1]
    return v / np.sum(v)

def extractF0(obsProbList):
    nHop = len(obsProbList)

    out = np.zeros(nHop, dtype = np.float64)
    for iHop, (freqProb) in enumerate(obsProbList):
        if(len(freqProb) > 0):
            out[iHop] = freqProb.T[0][np.argmax(freqProb.T[1])]

    return out

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

    def __call__(self, x):
        (nX,) = x.shape
        nHop = getNFrame(nX, self.hopSize)
        (pdfSize,) = self.pdf.shape

        if(self.prefilter):
            x = yin.applyPrefilter(x, self.maxFreq, self.samprate)

        out = []

        for iHop in range(nHop):
            iCenter = int(round(iHop * self.hopSize))
            windowSize = 0
            newWindowSize = int(max(self.samprate / self.minFreq * 4, self.minWindowSize) * self.windowLengthFac)
            if(newWindowSize % 2 == 1):
                newWindowSize += 1
            iIter = 0
            while(newWindowSize != windowSize and iIter < self.maxIter):
                windowSize = newWindowSize
                frame = getFrame(x, iCenter, windowSize)
                if(self.removeDC):
                    frame = removeDCSimple(frame)
                
                buff = yin.difference(frame)
                buff = yin.cumulativeDifference(buff)
                valleyIndexList = yin.findValleys(buff, self.minFreq, self.maxFreq, self.samprate, threshold = self.valleyThreshold, step = self.valleyStep)
                nValley = len(valleyIndexList)
                if(valleyIndexList):
                    possibleFreq = min(self.maxFreq, max(self.samprate / valleyIndexList[-1] - 20.0, self.minFreq))
                    newWindowSize = int(max(self.samprate / possibleFreq * 4, self.minWindowSize) * self.windowLengthFac)
                    if(newWindowSize % 2 == 1):
                        newWindowSize += 1
                    iIter += 1

            freqProb = np.zeros((nValley, 2), dtype = np.float64)
            probTotal = 0.0
            weightedProbTotal = 0.0
            for iValley, valley in enumerate(valleyIndexList):
                ipledIdx, ipledVal = parabolicInterpolate(buff, valley)
                freq = self.samprate / ipledIdx
                v0 = 1 if(iValley == 0) else min(1.0, buff[valleyIndexList[iValley - 1]] + eps)
                v1 = 0 if(iValley == nValley - 1) else max(0.0, buff[valleyIndexList[iValley + 1]]) + eps
                prob = 0.0
                for i in range(int(v1 * pdfSize), int(v0 * pdfSize)):
                    prob += self.pdf[i] * (1.0 if(ipledVal < i / pdfSize) else 0.01)
                prob = min(prob, 0.99)
                prob *= self.bias
                probTotal += prob
                if(ipledVal < self.probThreshold):
                    prob *= self.weightPrior
                weightedProbTotal += prob
                freqProb[iValley] = freq, prob

            # renormalize
            if(nValley > 0 and weightedProbTotal != 0.0):
                freqProb.T[1] *= probTotal / weightedProbTotal

            out.append(freqProb)

        return out
