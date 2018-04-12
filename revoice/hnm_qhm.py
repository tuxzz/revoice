from .common import *
from . import adaptivestft
from . import hnm

def hmToSinusoid(fkHat, ak):
    K = len(ak)
    nHar = (K - 1) // 2

    hAmp = np.zeros(nHar, dtype = np.float64)
    hPhase = np.zeros(nHar, dtype = np.float64)
    hFreq = fkHat[nHar + 1:].reshape(nHar)
    hAmp = np.abs(ak[nHar + 1:].reshape(nHar)) * 2.0
    hPhase = np.unwrap(np.angle(ak[nHar + 1:]).reshape(nHar))

    return hFreq, hAmp, hPhase

class Slover:
    def __init__(self, mvf, sr, **kwargs):
        self.samprate = float(sr)
        self.window = kwargs.get("window", "blackman")

        self.mvf = float(mvf)
        self.stepMaxCorrect = kwargs.get("stepMaxCorrect", 20.0)

        self.x = None

        self.f0 = None
        self.nHar = None

        self.K = None
        self.windowArray = None
        self.fkHat = None
        self.n = None
        self.windowedX = None
        self.ak = None
        self.bk = None

    def reset(self, x, f0, nHar):
        (nX,) = x.shape
        assert nX % 2 == 1

        self.x = x

        self.f0 = f0
        self.nHar = nHar

        self.K = nHar * 2 + 1
        N = (nX - 1) // 2
        self.windowArray = getWindow(self.window)[0](nX).reshape(nX, 1)
        self.fkHat = np.arange(-self.nHar, self.nHar + 1).reshape(self.K, 1) * self.f0
        self.n = np.arange(-N, N + 1).reshape(1, nX)
        self.windowedX = x.reshape(nX, 1) * self.windowArray

    def iterate(self):
        (nX,) = self.x.shape

        t = np.dot(2 * np.pi * self.fkHat, self.n) / self.samprate # arg of cplx exp
        E = np.cos(t) + 1j * np.sin(t) # mat with cplx exp, [2N + 1 * K] dim
        E = np.concatenate((E, np.tile(self.n, (self.K, 1)) * E), axis = 0) # (2K, 2N + 1)
        Ew = np.tile(self.windowArray, (1, 2 * self.K)) * E.T # multiply the window
        R = np.dot(Ew.T, Ew) # compute the matrix to be inverted
        theta = sla.lstsq(R, np.dot(Ew.T, self.windowedX))[0]
        self.ak = theta[:self.K] # cplx amps
        self.bk = theta[self.K:] # cplx slopes

    def correct(self):
        self.fkHat += np.clip(self.samprate / (2 * np.pi) * (self.ak.real * self.bk.imag - self.ak.imag * self.bk.real) / (np.abs(self.ak) ** 2), -self.stepMaxCorrect, self.stepMaxCorrect)
        self.fkHat = np.clip(self.fkHat, 50.0, self.samprate / 2)

class Analyzer:
    def __init__(self, hopSize, mvf, sr, **kwargs):
        self.samprate = sr
        self.mvf = mvf
        self.hopSize = hopSize
        self.window = kwargs.get("window", "blackman")
        
        self.maxAIRIter = kwargs.get("maxAIRIter", 10)
        self.maxQHMIter = kwargs.get("maxQHMIter", 0) # set it to a positive value for quasi-harmonic
        self.maxAirAvgHar = kwargs.get("maxAirAvgHar", 8)
        self.stepMaxCorrect = kwargs.get("stepMaxCorrect", 20.0)
        self.airSRERThreshold = kwargs.get("airSRERThreshold", 0.1)
        self.slover = Slover(self.mvf, self.samprate, window = self.window, stepMaxCorrect = self.stepMaxCorrect)
        assert self.mvf < self.samprate / 2

    def __call__(self, x, f0List, maxHar):
        # constant
        (nX,) = x.shape
        nHop = getNFrame(nX, self.hopSize)

        # check input
        assert (nHop,) == f0List.shape

        synthSize = int(self.hopSize * 2)
        if(synthSize % 2 == 1):
            synthSize += 1
        halfSynthSize = synthSize // 2

        f0List = f0List.copy()
        synthRange = np.arange(-halfSynthSize, halfSynthSize) / self.samprate
        synthWindow = np.hanning(synthSize)
        nVoiced = np.sum(f0List > 0.0)

        # AIR
        if(self.maxAIRIter > 0):
            srerSum = 0.0
            for iHop, f0 in enumerate(f0List):
                if(f0 <= 0.0):
                    continue
                iCenter = int(round(iHop * self.hopSize))
                comparisonFrame = getFrame(x, iCenter, synthSize) * synthWindow

                analyzeWindowSize = int(self.samprate / f0 * 4.0)
                if(analyzeWindowSize % 2 == 0):
                    analyzeWindowSize += 1
                analyzeFrame = getFrame(x, iCenter, analyzeWindowSize)
                nHar = min(int(self.mvf / f0), maxHar)
                airAvgHar = min(nHar, self.maxAirAvgHar)
                self.slover.reset(analyzeFrame, f0, nHar)
                self.slover.iterate()
                hFreq, hAmp, hPhase = hmToSinusoid(self.slover.fkHat, self.slover.ak)
                synthed = hnm.synthSinusoid(synthRange, hFreq, hAmp, hPhase, self.samprate) * synthWindow
                lastSRER = calcSRER(comparisonFrame, synthed)
                lastF0 = f0

                for iIter in range(self.maxAIRIter):
                    self.slover.correct()
                    f0 = np.mean(self.slover.fkHat[nHar + 1:nHar + 1 + airAvgHar].reshape(airAvgHar) / np.arange(1, airAvgHar + 1))
                    analyzeWindowSize = int(self.samprate / lastF0 * 4.0)
                    if(analyzeWindowSize % 2 == 0):
                        analyzeWindowSize += 1
                    analyzeFrame = getFrame(x, iCenter, analyzeWindowSize)
                    nHar = int(self.mvf / lastF0)
                    airAvgHar = min(nHar, self.maxAirAvgHar)
                    self.slover.reset(analyzeFrame, lastF0, nHar)
                    self.slover.iterate()
                    hFreq, hAmp, hPhase = hmToSinusoid(self.slover.fkHat, self.slover.ak)
                    synthed = hnm.synthSinusoid(synthRange, hFreq, hAmp, hPhase, self.samprate) * synthWindow
                    srer = calcSRER(comparisonFrame, synthed)
                    if(srer - lastSRER > 0.0):
                        lastF0 = f0
                        lastSRER = srer
                    if(srer - lastSRER < self.airSRERThreshold):
                        break
                f0List[iHop] = lastF0
                srerSum += lastSRER
        print("AIR Average SRER:", srerSum / nVoiced)

        # QHM
        minF0 = np.min(f0List[f0List > 0.0])
        srerSum = 0.0
        hFreqList = np.zeros((nHop, maxHar), dtype = np.float64)
        hAmpList = np.zeros((nHop, maxHar), dtype = np.float64)
        hPhaseList = np.zeros((nHop, maxHar), dtype = np.float64)

        for iHop, f0 in enumerate(f0List):
            if(f0 <= 0.0):
                continue
            iCenter = int(round(iHop * self.hopSize))

            comparisonFrame = getFrame(x, iCenter, synthSize) * synthWindow
            analyzeWindowSize = int(self.samprate / f0 * 4.0)
            if(analyzeWindowSize % 2 == 0):
                analyzeWindowSize += 1
            analyzeFrame = getFrame(x, iCenter, analyzeWindowSize)
            nHar = min(int(self.mvf / f0), maxHar)
            self.slover.reset(analyzeFrame, f0, nHar)
            self.slover.iterate()
            lastHFreq, lastHAmp, lastHPhase = hmToSinusoid(self.slover.fkHat, self.slover.ak)
            synthed = hnm.synthSinusoid(synthRange, lastHFreq, lastHAmp, lastHPhase, self.samprate) * synthWindow
            lastSRER = calcSRER(comparisonFrame, synthed)

            for iIter in range(self.maxQHMIter):
                self.slover.correct()
                self.slover.iterate()
                hFreq, hAmp, hPhase = hmToSinusoid(self.slover.fkHat, self.slover.ak)
                synthed = hnm.synthSinusoid(synthRange, hFreq, hAmp, hPhase, self.samprate) * synthWindow
                srer = calcSRER(comparisonFrame, synthed)
                if(srer - lastSRER > 0.0):
                    lastHFreq, lastHAmp, lastHPhase = hFreq, hAmp, hPhase
                    lastSRER = srer
                else:
                    break
            need = np.logical_and(lastHFreq > 0.0, lastHFreq < self.mvf)
            lastHFreq, lastHAmp, lastHPhase = lastHFreq[need], lastHAmp[need], lastHPhase[need]
            (nHar,) = lastHFreq.shape
            order = np.argsort(lastHFreq)
            hFreqList[iHop, :nHar], hAmpList[iHop, :nHar], hPhaseList[iHop, :nHar] = lastHFreq[order], lastHAmp[order], lastHPhase[order]
            srerSum += lastSRER
        print("QHM Average SRER:", srerSum / nVoiced)
        return hFreqList, hAmpList, hPhaseList
