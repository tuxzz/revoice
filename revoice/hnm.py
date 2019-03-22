from .common import *
from . import cheaptrick, adaptivestft, hnm_qfft, hnm_get, hnm_qhm

@nb.jit(nb.float32[:](nb.float32[:], nb.float32[:], nb.float32[:], nb.float32[:], nb.float32), parallel=True, fastmath=True, nopython=True)
def synthSinusoid(t, hFreq, hAmp, hPhase, sr):
    # constant
    nOut = t.shape[0]
    nSin = hFreq.shape[0]
    nyq = sr / 2

    # check input
    assert nSin == hAmp.shape[0]
    assert nSin == hPhase.shape[0]
    assert hFreq.ndim == 1
    assert hAmp.ndim == 1
    assert hPhase.ndim == 1

    # compute
    out = np.zeros(nOut, dtype=np.float32)
    for iSin in nb.prange(nSin):
        freq = hFreq[iSin]
        amp = hAmp[iSin]
        phase = hPhase[iSin] if(hPhase is not None) else 0.0
        if freq <= 0.0 or freq >= nyq:
            break
        if amp <= 0.0:
            continue
        out[:] += np.cos(2.0 * np.pi * freq * t + phase) * amp
    return out

_ndc_a = sigmoid(-5.0)
_ndc_b = 1.0 / (1.0 - sigmoid(-5.0))
def noiseDCFunction(snr):
    return 1.0 - (sigmoid(snr / 50.0 - 5.0) - _ndc_a) * _ndc_b * 0.95

def calculateSinusoidSpectrum(hFreq, hAmp, hPhase, nFFT, sr):
    synthRange = np.arange(4096) / sr
    window = np.hanning(4096)
    windowNormFac = 2 / np.sum(window)
    return np.fft.rfft(synthSinusoid(synthRange, hFreq, hAmp, hPhase, sr)).astype(np.float32) * windowNormFac

def filterNoise(x, responseList, hopSize):
    olaFac = 2
    windowFac = 4
    nHop, responseBin = responseList.shape

    windowSize = int(hopSize * windowFac)
    if windowSize % 2 == 1:
        windowSize += 1
    
    nBin = windowSize // 2 + 1
    nX = x.shape[0]

    window = np.hanning(windowSize).astype(np.float32)
    analyzeNormFac = 0.5 * np.sum(window)
    synthNormScale = windowFac // 2 * olaFac

    window = np.sqrt(window)
    buff = np.zeros(nBin, dtype=np.complex64)
    out = np.zeros(nX, dtype=np.float32)
    for iFrame in range(nHop * olaFac):
        iHop = iFrame // olaFac
        iCenter = int(round(iFrame * hopSize / olaFac))
        
        frame = removeDCSimple(getFrame(x, iCenter, windowSize))
        if np.max(frame) == np.min(frame):
            continue

        ffted = np.fft.rfft(frame * window)
        phase = np.angle(ffted)

        env = ipl.interp1d(np.linspace(0, nBin, responseBin), responseList[iHop], kind = "linear")(np.arange(nBin))
        magn = np.exp(env) * analyzeNormFac

        buff.real = magn * np.cos(phase)
        buff.imag = magn * np.sin(phase)

        synthed = np.fft.irfft(buff) * window
        ob, oe, ib, ie = getFrameRange(nX, iCenter, windowSize)
        out[ib:ie] += synthed[ob:oe] / synthNormScale
    return out

class Analyzer:
    supportedHarmonicAnalyzer = {
        "qfft": hnm_qfft.Analyzer,
        "get": hnm_get.Analyzer,
        "qhm": hnm_qhm.Analyzer,
    }

    def __init__(self, sr, **kwargs):
        self.samprate = float(sr)
        self.hopSize = kwargs.get("hopSize", self.samprate * 0.0025)
        self.fftSize = kwargs.get("fftSize", roundUpToPowerOf2(self.samprate * 0.05))

        self.mvf = kwargs.get("mvf", min(sr / 2 - 1e3, 20e3))
        self.harmonicAnalysisMethod = kwargs.get("harmonicAnalysisMethod", "qfft")
        self.harmonicAnalyzerParameter = kwargs.get("harmonicAnalyzerParameter", {})
        self.noiseEnergyThreshold = kwargs.get("noiseEnergyThreshold", 1e-8)

        nyq = self.samprate / 2
        assert self.samprate > 0
        assert self.mvf > 0 and self.mvf <= nyq
        assert isinstance(self.fftSize, int)

    def __call__(self, x, f0List):
        # early check input
        assert f0List.shape[0] > 0

        # constant
        (nX,) = x.shape
        nBin = self.fftSize // 2 + 1
        nHop = getNFrame(nX, self.hopSize)
        minF0 = np.min(f0List[f0List > 0.0])
        maxHar = max(0, int(self.mvf / minF0))

        synthSize = int(self.hopSize * 2)
        if synthSize % 2 == 1:
            synthSize += 1
        halfSynthSize = synthSize // 2

        # check input
        assert f0List.shape == (nHop,)
        assert self.fftSize == roundUpToPowerOf2(self.fftSize)

        # (quasi) harmonic analysis
        harmonicAnalyzer = self.supportedHarmonicAnalyzer[self.harmonicAnalysisMethod](self.hopSize, self.mvf, self.samprate, **self.harmonicAnalyzerParameter)
        hFreqList, hAmpList, hPhaseList = harmonicAnalyzer(x, f0List, maxHar)
        hF0List = hFreqList[:, 0]
        hMeanF0 = np.mean(hF0List[hF0List > 0])

        del harmonicAnalyzer

        # resynth & ola & calculate sinusoid energy
        sinusoid = np.zeros(nX, dtype=np.float32)
        sinusoidEnergyList = np.zeros(nHop, dtype=np.float32)
        olaWindow = np.hanning(synthSize).astype(np.float32)
        for iHop, f0 in enumerate(f0List):
            if(f0 <= 0.0):
                continue
            iCenter = int(round(iHop * self.hopSize))
            energyAnalysisRadius = int(self.samprate / f0 * 2)
            if(energyAnalysisRadius % 2 == 1):
                energyAnalysisRadius += 1
            
            synthLeft = max(energyAnalysisRadius, halfSynthSize)
            synthRight = max(energyAnalysisRadius + 1, halfSynthSize)
            synthRange = np.arange(-synthLeft, synthRight, dtype=np.float32) / self.samprate

            ob, oe, ib, ie = getFrameRange(nX, iCenter, synthSize)
            synthed = synthSinusoid(synthRange, hFreqList[iHop], hAmpList[iHop], hPhaseList[iHop], self.samprate)

            # integrate energy
            energyBegin = synthLeft - energyAnalysisRadius
            energyAnalysisWindow = np.hanning(energyAnalysisRadius * 2 + 1).astype(np.float32)
            energyAnalysisWindowNormFac = 1 / np.mean(energyAnalysisWindow)
            sinusoidEnergyList[iHop] = np.mean((synthed[energyBegin:energyBegin + energyAnalysisRadius * 2 + 1] * energyAnalysisWindow * energyAnalysisWindowNormFac) ** 2)

            # ola
            olaBegin = synthLeft - halfSynthSize
            sinusoid[ib:ie] += synthed[olaBegin + ob:olaBegin + oe] * olaWindow[ob:oe]
            del energyAnalysisRadius, synthLeft, synthRight, synthRange, ob, oe, ib, ie, synthed, energyBegin, energyAnalysisWindow, olaBegin
        del olaWindow

        # extract noise by subtract
        noise = x - sinusoid

        # build noise envelope
        noiseMagnList = np.abs(adaptivestft.Analyzer(self.samprate)(noise, hF0List))
        noiseEnvList = cheaptrick.Analyzer(self.samprate, fixedF0=hMeanF0)(noiseMagnList, hF0List)
        del noiseMagnList

        # calculate noise energy
        noiseEnergyList = np.zeros(nHop, dtype=np.float32)
        for iHop, f0 in enumerate(f0List):
            iCenter = int(round(iHop * self.hopSize))
            if f0 > 0.0:
                frame = getFrame(noise, iCenter, 4 * int(self.samprate / f0) + 1)
            else:
                frame = getFrame(noise, iCenter, synthSize)
            energyAnalysisWindow = np.hanning(frame.shape[0])
            energyAnalysisWindowNormFac = 1 / np.mean(energyAnalysisWindow)
            noiseEnergyList[iHop] = np.mean((frame * energyAnalysisWindow * energyAnalysisWindowNormFac) ** 2)
        noiseEnergyList[noiseEnergyList < self.noiseEnergyThreshold] = 0.0
        
        # debug
        saveWav("debug/sin.wav", sinusoid, self.samprate)
        saveWav("debug/noise.wav", noise, self.samprate)

        return hFreqList, hAmpList, hPhaseList, sinusoidEnergyList, noiseEnvList, noiseEnergyList

class Synther:
    def __init__(self, sr, **kwargs):
        self.samprate = float(sr)
        self.hopSize = kwargs.get("hopSize", self.samprate * 0.0025)
        self.fftSize = kwargs.get("fftSize", roundUpToPowerOf2(self.samprate * 0.05))

        self.mvf = kwargs.get("mvf", min(sr / 2 - 1e3, 20e3))
        self.maxNoiseEnvHarmonic = kwargs.get("maxNoiseEnvHarmonic", 3)
        self.maxNoiseEnvDCAdjustment = kwargs.get("maxNoiseEnvDCAdjustment", 10.0)

        assert self.mvf <= self.samprate / 2
        assert isinstance(self.fftSize, int)

    def __call__(self, hFreqList, hAmpList, hPhaseList, sinusoidEnergyList, noiseEnvList, noiseEnergyList, enableSinusoid = True, enableNoise = True):
        # early check input
        assert hFreqList.ndim == 2
        # constant
        nHop, nHar = hFreqList.shape
        nOut = getNSample(nHop, self.hopSize)
        nBin = self.fftSize // 2 + 1

        # check input
        assert hAmpList.shape == (nHop, nHar)
        assert hPhaseList.shape == (nHop, nHar)
        assert sinusoidEnergyList is None or sinusoidEnergyList.shape == (nHop,)
        assert not enableNoise or noiseEnvList.shape == (nHop, nBin)
        assert not enableNoise or (noiseEnergyList is None or noiseEnergyList.shape == (nHop,))

        synthSize = int(self.hopSize * 2)
        if(synthSize % 2 == 1):
            synthSize += 1
        halfSynthSize = synthSize // 2

        # synth sinusoid
        if enableSinusoid:
            sinusoid = np.zeros(nOut, dtype=np.float32)
            synthWindow = np.hanning(synthSize).astype(np.float32)
            for iHop in range(nHop):
                f0 = hFreqList[iHop, 0]
                if f0 <= 0.0 or (sinusoidEnergyList is not None and sinusoidEnergyList[iHop] <= 0.0):
                    continue
                iCenter = int(round(iHop * self.hopSize))

                need = hFreqList[iHop] > 0.0
                hFreq = hFreqList[iHop][need]
                hAmp = hAmpList[iHop][need]
                hPhase = hPhaseList[iHop][need]
                if sinusoidEnergyList is None:
                    synthRange = np.arange(-halfSynthSize, halfSynthSize, dtpye=np.float32) / self.samprate
                    synthed = synthSinusoid(synthRange, hFreq, hAmp, hPhase, self.samprate)
                    ob, oe, ib, ie = getFrameRange(nOut, iCenter, synthSize)
                    sinusoid[ib:ie] += synthed[ob:oe] * synthWindow[ob:oe]
                    del synthRange, synthed, ob, oe, ib, ie
                else:
                    energyAnalysisRadius = int(self.samprate / f0 * 2)
                    if(energyAnalysisRadius % 2 == 1):
                        energyAnalysisRadius += 1
                    
                    synthLeft = max(energyAnalysisRadius, halfSynthSize)
                    synthRight = max(energyAnalysisRadius + 1, halfSynthSize)
                    synthRange = np.arange(-synthLeft, synthRight, dtype=np.float32) / self.samprate
                    energyAnalysisWindow = np.hanning(energyAnalysisRadius * 2 + 1).astype(np.float32)
                    energyAnalysisWindowNormFac = 1 / np.mean(energyAnalysisWindow)

                    synthed = synthSinusoid(synthRange, hFreq, hAmp, hPhase, self.samprate)

                    # integrate energy
                    energyAnalysisBegin = synthLeft - energyAnalysisRadius
                    energyAnalysisWindow = np.hanning(energyAnalysisRadius * 2 + 1)
                    synthedEnergy = np.mean((synthed[energyAnalysisBegin:energyAnalysisBegin + energyAnalysisRadius * 2 + 1] * energyAnalysisWindow * energyAnalysisWindowNormFac) ** 2)

                    # ola
                    olaBegin = synthLeft - halfSynthSize
                    ob, oe, ib, ie = getFrameRange(nOut, iCenter, synthSize)

                    sinusoidEnergy = sinusoidEnergyList[iHop]
                    sinusoid[ib:ie] += synthed[olaBegin + ob:olaBegin + oe] * np.sqrt(sinusoidEnergy / synthedEnergy) * synthWindow[ob:oe]
                    del iHop, f0, energyAnalysisRadius, synthLeft, synthRight, synthRange, energyAnalysisWindow, energyAnalysisWindowNormFac
                    del need, synthed, energyAnalysisBegin, synthedEnergy
                    del olaBegin, ob, oe, ib, ie, sinusoidEnergy
            del synthWindow
        
        # synth noise
        if enableNoise:
            noise = np.zeros(nOut, dtype=np.float32)
            noiseTemplate = np.random.uniform(-1.0, 1.0, nOut).astype(np.float32)
            noiseTemplate = filterNoise(noiseTemplate, noiseEnvList, self.hopSize)
            synthWindow = np.hanning(synthSize).astype(np.float32)

            # energy normalize & apply noise energy envelope
            for iHop in range(nHop):
                if noiseEnergyList[iHop] <= 0.0:
                    continue
                iCenter = int(round(iHop * self.hopSize))

                f0 = hFreqList[iHop, 0]
                noiseEnergy = noiseEnergyList[iHop]

                # synth noise energy envelope from harmonic if f0 greater than 0
                if sinusoidEnergyList is not None and f0 > 0.0:
                    sinusoidEnergy = sinusoidEnergyList[iHop]
                    
                    energyAnalysisRadius = int(self.samprate / f0 * 2)
                    if energyAnalysisRadius % 2 == 1:
                        energyAnalysisRadius += 1
                    
                    synthLeft = max(energyAnalysisRadius, halfSynthSize)
                    synthRight = max(energyAnalysisRadius + 1, halfSynthSize)
                    synthRange = np.arange(-synthLeft, synthRight, dtype=np.float32) / self.samprate

                    need = hFreqList[iHop] > 0.0
                    nHar = min(np.sum(need), self.maxNoiseEnvHarmonic)
                    synthedNoiseEnergyEnv = synthSinusoid(synthRange, hFreqList[iHop][need][:nHar], hAmpList[iHop][need][:nHar], hPhaseList[iHop][need][:nHar], self.samprate)
                    
                    # set as positive
                    synthedNoiseEnergyEnv -= np.min(synthedNoiseEnergyEnv)
                    synthedNoiseEnergyEnvNormFac = np.max(synthedNoiseEnergyEnv)
                    if synthedNoiseEnergyEnvNormFac > 0.0:
                        synthedNoiseEnergyEnv /= synthedNoiseEnergyEnvNormFac
                    # dc adjustment
                    if noiseEnergy > 0.0:
                        snr = sinusoidEnergy / noiseEnergy
                        dc = noiseDCFunction(snr)
                        synthedNoiseEnergyEnv *= 1.0 - dc
                        synthedNoiseEnergyEnv += dc

                    # apply energy envelope to template
                    noiseTemplateFrame = getFrame(noiseTemplate, iCenter, synthRight + synthLeft)
                    noiseTemplateFrame *= synthedNoiseEnergyEnv
                    
                    # integrate template energy
                    energyAnalysisBegin = synthLeft - energyAnalysisRadius
                    energyAnalysisWindow = np.hanning(energyAnalysisRadius * 2 + 1).astype(np.float32)
                    energyAnalysisWindowNormFac = 1 / np.mean(energyAnalysisWindow)
                    noiseTemplateEnergy = np.mean((noiseTemplateFrame[energyAnalysisBegin:energyAnalysisBegin + energyAnalysisRadius * 2 + 1] * energyAnalysisWindow * energyAnalysisWindowNormFac) ** 2)

                    # set energy and ola
                    olaBegin = synthLeft - halfSynthSize
                    if noiseTemplateEnergy > 0.0:
                        windowedNoiseFrame = noiseTemplateFrame[olaBegin:olaBegin + synthSize] * np.sqrt(noiseEnergy / noiseTemplateEnergy) * synthWindow
                    else:
                        windowedNoiseFrame = np.zeros(synthSize, dtype=np.float32)
                    ob, oe, ib, ie = getFrameRange(nOut, iCenter, synthSize)
                    noise[ib:ie] += windowedNoiseFrame[ob:oe]
                else:
                    noiseTemplateFrame = getFrame(noiseTemplate, iCenter, synthSize)
                    # integrate template energy
                    if noiseEnergyList is not None:
                        energyAnalysisWindow = np.hanning(synthSize).astype(np.float32)
                        energyAnalysisWindowNormFac = 1 / np.mean(energyAnalysisWindow)
                        noiseTemplateEnergy = np.mean((noiseTemplateFrame * energyAnalysisWindow * energyAnalysisWindowNormFac) ** 2)
                        # set energy
                        if noiseTemplateEnergy > 0.0:
                            windowedNoiseFrame = noiseTemplateFrame * np.sqrt(noiseEnergy / noiseTemplateEnergy) * synthWindow
                        else:
                            windowedNoiseFrame = np.zeros(synthSize, dtype=np.float32)
                    else:
                        windowedNoiseFrame = noiseTemplateFrame * synthWindow
                    # ola
                    ob, oe, ib, ie = getFrameRange(nOut, iCenter, synthSize)
                    noise[ib:ie] += windowedNoiseFrame[ob:oe]
        
        # combine and output debug
        out = np.zeros(nOut, dtype=np.float32)
        if enableSinusoid:
            saveWav("debug/rs_sin.wav", sinusoid, self.samprate)
            out += sinusoid
        if enableNoise:
            saveWav("debug/rs_noise.wav", noise, self.samprate)
            out += noise
        saveWav("debug/rs_combine.wav", out, self.samprate)

        return out