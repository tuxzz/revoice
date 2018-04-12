from .common import *
from . import sparsehmm

def parameterFromPYin(pyin):
    hopSize = pyin.hopSize
    samprate = pyin.samprate
    nSemitone = int(np.ceil(np.log2(pyin.maxFreq / pyin.minFreq) * 12.0))
    maxTransSemitone = (pyin.hopSize / pyin.samprate) / 0.0055 * 3.0
    minFreq = pyin.minFreq
    minWindowSize = pyin.minWindowSize
    return hopSize, samprate, nSemitone, maxTransSemitone, minFreq, minWindowSize

class Analyzer:
    def __init__(self, hopSize, samprate, nSemitone, maxTransSemitone, minFreq, minWindowSize, **kwargs):
        self.hopSize = hopSize
        self.samprate = samprate
        self.nSemitone = nSemitone
        self.maxTransSemitone = maxTransSemitone
        self.minFreq = minFreq
        self.minWindowSize = minWindowSize
        self.binPerSemitone = kwargs.get("binPerSemitone", 5)
        self.transSelf = kwargs.get("transSelf", 0.999)
        self.inputTrust = kwargs.get("inputTrust", 0.5)

        self.model, self.initStateProb, self.sourceState, self.targetState, self.stateTransProb = self.createModel()

    def createModel(self):
        nBin = int(self.nSemitone * self.binPerSemitone)
        halfMaxTransBin = int(round((self.maxTransSemitone * self.binPerSemitone) / 2))
        nState = 2 * nBin
        nTrans = 4 * (nBin * (2 * halfMaxTransBin + 1) - halfMaxTransBin * (halfMaxTransBin + 1))
        init = np.ndarray(nState, dtype = np.float64)
        frm = np.zeros(nTrans, dtype = np.int)
        to = np.zeros(nTrans, dtype = np.int)
        transProb = np.zeros(nTrans, dtype = np.float64)

        init.fill(1.0 / nState)
        iA = 0
        for iBin in range(nBin):
            theoreticalMinNextBin = iBin - halfMaxTransBin
            minNextBin = max(iBin - halfMaxTransBin, 0)
            maxNextBin = min(iBin + halfMaxTransBin, nBin - 1)

            weights = np.zeros((maxNextBin - minNextBin + 1), dtype = np.float64)

            for i in range(minNextBin, maxNextBin + 1):
                if(i <= iBin):
                    weights[i - minNextBin] = i - theoreticalMinNextBin + 1.0
                else:
                    weights[i - minNextBin] = iBin - theoreticalMinNextBin + 1.0 - (i - iBin)
            weightSum = np.sum(weights)

            # trans to close pitch
            for i in range(minNextBin, maxNextBin + 1):
                frm[iA] = iBin
                to[iA] = i
                transProb[iA] = weights[i - minNextBin] / weightSum * self.transSelf

                frm[iA + 1] = iBin
                to[iA + 1] = i + nBin
                transProb[iA + 1] = weights[i - minNextBin] / weightSum * (1.0 - self.transSelf)

                frm[iA + 2] = iBin + nBin
                to[iA + 2] = i + nBin
                transProb[iA + 2] = weights[i - minNextBin] / weightSum * self.transSelf

                frm[iA + 3] = iBin + nBin
                to[iA + 3] = i
                transProb[iA + 3] = weights[i - minNextBin] / weightSum * (1.0 - self.transSelf)
                iA += 4

        return sparsehmm.ViterbiDecoder(nState, nTrans), init, frm, to, transProb

    def calcStateProb(self, obsProb):
        nBin = int(self.nSemitone * self.binPerSemitone)
        nState = self.model.nState
        maxFreq = self.minFreq * np.power(2, self.nSemitone / 12)
        probYinPitched = 0.0

        out = np.zeros(nState, dtype = np.float64)
        for freq, prob in obsProb:
            if(freq < self.minFreq or freq > maxFreq):
                if(freq <= 0.0):
                    break
                continue
            iBin = min(nBin - 1, max(0, int(round(np.log2(freq / self.minFreq) * 12 * self.binPerSemitone))))
            out[iBin] = prob
            probYinPitched += prob

        probReallyPitched = self.inputTrust * probYinPitched

        if(probYinPitched > 0.0):
            out[:nBin] *= probReallyPitched / probYinPitched
        out[nBin:] = (1.0 - probReallyPitched) / nBin
        out = np.clip(out, 0.0, np.inf) + 1e-05
        return out

    def __call__(self, obsProbList):
        # constant
        nBin = int(self.nSemitone * self.binPerSemitone)
        nState = self.model.nState
        maxFreq = self.minFreq * np.power(2, self.nSemitone / 12)
        nHop = len(obsProbList)

        # check input
        assert(nHop == len(obsProbList))
        if(isinstance(obsProbList, list)):
            pass
        elif(isinstance(obsProbList, np.ndarray)):
            assert(obsProbList.ndim == 2)
            assert(obsProbList.shape[1] == 2)
        else:
            raise TypeError("Unsupported obsSeq type")

        # feed & decode
        self.model.initialize(self.calcStateProb(obsProbList[0]), self.initStateProb)
        for iHop in range(1, nHop):
            self.model.feed(self.calcStateProb(obsProbList[iHop]), self.sourceState, self.targetState, self.stateTransProb)
        path = self.model.readDecodedPath()
        self.model.finalize()

        # extract frequency from path
        out = np.zeros(nHop, dtype = np.float64)
        for iHop in range(nHop):
            if(path[iHop] < nBin):
                hmmFreq = self.minFreq * np.power(2, path[iHop] / (12.0 * self.binPerSemitone))
                if(len(obsProbList[iHop]) == 0):
                    bestFreq = hmmFreq
                else:
                    iNearest = np.argmin(np.abs(obsProbList[iHop].T[0] - hmmFreq))
                    bestFreq = obsProbList[iHop][iNearest][0]
                    if(bestFreq < self.minFreq or bestFreq > maxFreq or abs(np.log2(bestFreq / self.minFreq) * 12 * self.binPerSemitone - path[iHop]) > 1.0):
                        bestFreq = hmmFreq
            else:
                bestFreq = -self.minFreq * np.power(2, (path[iHop] - nBin) / (12 * self.binPerSemitone))
            out[iHop] = bestFreq

        # mark unvoiced->voiced bound as voiced
        for iHop in range(1, nHop):
            if(out[iHop - 1] <= 0.0 and out[iHop] > 0.0):
                windowSize = min(self.samprate / out[iHop] * 4, self.minWindowSize)
                if(windowSize % 2 == 1):
                    windowSize += 1
                frameOffset = int(round(windowSize / self.minWindowSize))
                out[max(0, iHop - frameOffset):iHop] = out[iHop]

        return out
