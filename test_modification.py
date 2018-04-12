import numpy as np
import pylab as pl
from revoice import *
from revoice.common import *
import os, pickle

path = "./voices/yuri_orig.wav"

timeFac = 3.0
pitchFac = 0.75
fixedPitch = None
rdFac = 1.0
hnmAnalysisMethod = "get"
cachePath = "%s.%s.%s" % (path, hnmAnalysisMethod, "cache.pickle")

if(os.path.isfile(cachePath)):
    print("Load cache...")
    with open(cachePath, "rb") as f:
        w, sr, energyList, hFreqList, hAmpList, hPhaseList, sinusoidEnergyList, noiseEnvList, noiseEnergyList, hopSize, nHop = pickle.load(f)
else:
    print("Load file...")
    w, sr = loadWav(path)

    print("Energy Analyzing...")
    energyList = energy.Analyzer(sr)(w)

    hopSize = energy.Analyzer(sr).hopSize
    nHop = energyList.shape[0]

    print("F0 Estimation...")
    pyinAnalyzer = pyin.Analyzer(sr)
    obsProbList = pyinAnalyzer(w)
    monopitchAnalyzer = monopitch.Analyzer(*monopitch.parameterFromPYin(pyinAnalyzer))
    f0List = monopitchAnalyzer(obsProbList)
    silentList = energyList < 1e-8
    f0List[silentList] = -np.abs(f0List[silentList])

    # fix vuv flag for voices/yuri_orig.wav
    if(path == "./voices/yuri_orig.wav"):
        f0List[int(0.19 * sr / pyinAnalyzer.hopSize):int(0.354 * sr / pyinAnalyzer.hopSize) + 1] = np.abs(f0List[int(0.19 * sr / pyinAnalyzer.hopSize):int(0.354 * sr / pyinAnalyzer.hopSize) + 1])
        f0List[int(2.669 * sr / pyinAnalyzer.hopSize):int(2.689 * sr / pyinAnalyzer.hopSize) + 1] = np.abs(f0List[int(2.669 * sr / pyinAnalyzer.hopSize):int(2.689 * sr / pyinAnalyzer.hopSize) + 1])
        f0List[int(2.814 * sr / pyinAnalyzer.hopSize):int(2.86 * sr / pyinAnalyzer.hopSize) + 1] = np.abs(f0List[int(2.814 * sr / pyinAnalyzer.hopSize):int(2.86 * sr / pyinAnalyzer.hopSize) + 1])

    print("F0 Refinement...")
    f0RefineProcessor = refinef0_stft.Processor(sr)
    f0List = f0RefineProcessor(w, f0List)

    print("HNM Analyzing...")
    hnmAnalyzer = hnm.Analyzer(sr, harmonicAnalysisMethod = hnmAnalysisMethod)
    hFreqList, hAmpList, hPhaseList, sinusoidEnergyList, noiseEnvList, noiseEnergyList = hnmAnalyzer(w, f0List)

    with open(cachePath, "wb") as f:
        pickle.dump((w, sr, energyList, hFreqList, hAmpList, hPhaseList, sinusoidEnergyList, noiseEnvList, noiseEnergyList, hopSize, nHop), f)

f0List = hFreqList[:, 0]

print("Rd Analyzing...")
nHop = f0List.shape[0]
rdList = np.zeros(nHop)
rdAnalyzer = rd_krh.Analyzer()
for iHop, f0 in enumerate(f0List):
    if(f0 <= 0):
        continue
    need = hFreqList[iHop] > 0
    rdList[iHop] = rdAnalyzer(hFreqList[iHop][need], hAmpList[iHop][need])

print("Voice tract Analyzing...")
hVtAmpList = np.zeros(hAmpList.shape)
hVsPhaseList = np.zeros(hPhaseList.shape)
for iHop, f0 in enumerate(f0List):
    if(f0 <= 0):
        continue
    need = hFreqList[iHop] > 0
    hFreq = hFreqList[iHop][need]
    hAmp = hAmpList[iHop][need]
    hPhase = hPhaseList[iHop][need]

    hVsAmp = np.abs(lfmodel.calcSpectrum(np.asarray(hFreq), 1 / f0, 1.0, *lfmodel.calcParameterFromRd(rdList[iHop])))
    hVsAmp /= hVsAmp[0]
    hVtAmpList[iHop][need] = hAmp / hVsAmp
    hVtAmp = hVtAmpList[iHop][need]
    hVtPhase = calcSinusoidMinphase(hFreq, hVtAmp)
    hVsPhaseList[iHop][need] = wrap(hPhase - hVtPhase)

print("HNM Sinusoid Synthing...")
synProc = hnm.Synther(sr)
sinusoid = synProc(hFreqList, hVtAmpList, hPhaseList, None, None, None, enableNoise = False)

print("Envelope Analyzing...")
envAnalyzer = mfienvelope.Analyzer(sr)
envList = envAnalyzer(sinusoid, f0List)

print("Inverse Phase Propagate for Voice Source...")
tList = np.arange(nHop) * hopSize / sr
hVsPhaseList = propagatePhase(hFreqList, hVsPhaseList, hopSize, sr, True)
hVsPhaseList = np.unwrap(hVsPhaseList, axis = 0)

print("Rd Shifting...")
rdList *= rdFac

print("Pitch Shifting...")
hFreqList *= pitchFac
if(fixedPitch):
    for iHop, f0 in enumerate(f0List):
        if(f0 <= 0):
            continue
        need = hFreqList[iHop] > 0
        hFreqList[iHop][need] = np.arange(1, np.sum(need) + 1) * fixedPitch
hFreqList[hFreqList >= sr / 2] = 0.0

print("Time Shifting...")
splittedF0List = splitArray(f0List)
nNewHop = int(nHop * timeFac)

# These are everything we need
newRdList = np.zeros(nNewHop)
newEnvList = np.zeros((nNewHop, envList.shape[1]))
newF0List = np.zeros(nNewHop)
newHVsPhaseList = np.zeros((nNewHop, hVsPhaseList.shape[1]))
newSinusoidEnergyList = np.zeros(nNewHop)
newNoiseEnvList = ipl.interp1d(np.arange(nHop), noiseEnvList, kind = "linear", bounds_error = True, axis = 0)(np.linspace(0, nHop - 1, nNewHop))
newNoiseEnergyList = ipl.interp1d(np.arange(nHop), noiseEnergyList, kind = "linear", bounds_error = True, axis = 0)(np.linspace(0, nHop - 1, nNewHop))

iHop = 0
for segment in splittedF0List:
    segmentSize = len(segment)
    if(segment[0] <= 0):
        iHop += segmentSize
        continue
    hopEnd = iHop + segmentSize
    iNewHop = int(iHop * timeFac)
    newHopEnd = int(hopEnd * timeFac)

    hopMapSource = np.arange(iHop, hopEnd)
    hopMap = np.linspace(iHop, hopEnd - 1, newHopEnd - iNewHop)

    newRdList[iNewHop:newHopEnd] = ipl.interp1d(hopMapSource, rdList[iHop:hopEnd], kind = "linear", bounds_error = True, axis = 0)(hopMap)
    newEnvList[iNewHop:newHopEnd] = ipl.interp1d(hopMapSource, envList[iHop:hopEnd], kind = "linear", bounds_error = True, axis = 0)(hopMap)
    newF0List[iNewHop:newHopEnd] = ipl.interp1d(hopMapSource, f0List[iHop:hopEnd], kind = "linear", bounds_error = True, axis = 0)(hopMap)
    newHVsPhaseList[iNewHop:newHopEnd] = ipl.interp1d(hopMapSource, hVsPhaseList[iHop:hopEnd], kind = "linear", bounds_error = True, axis = 0)(hopMap)
    newSinusoidEnergyList[iNewHop:newHopEnd] = ipl.interp1d(hopMapSource, sinusoidEnergyList[iHop:hopEnd], kind = "linear", bounds_error = True, axis = 0)(hopMap)

    iHop += segmentSize

# regenerate harmonic frequency
newHFreqList = newF0List.reshape(nNewHop, 1) * np.arange(1, hFreqList.shape[1] + 1)
newHFreqList[np.logical_or(newHFreqList < 0.0, newHFreqList >= sr / 2)] = 0.0

rdList = newRdList
envList = newEnvList
hFreqList = newHFreqList
hVsPhaseList = newHVsPhaseList
sinusoidEnergyList = newSinusoidEnergyList
noiseEnvList = newNoiseEnvList
noiseEnergyList = newNoiseEnergyList
nHop = nNewHop
f0List = hFreqList[:, 0]

hVtAmpList = np.zeros(hFreqList.shape)
hAmpList = np.zeros(hFreqList.shape)
hPhaseList = np.zeros(hFreqList.shape)

print("Recovery Voice Tract Amplitude From Envelope...")
for iHop, f0 in enumerate(f0List):
    if(f0 <= 0):
        continue
    need = hFreqList[iHop] > 0
    hFreq = hFreqList[iHop][need]
    hVtAmpList[iHop][need] = np.exp(ipl.interp1d(rfftFreq(roundUpToPowerOf2(0.05 * sr), sr), envList[iHop], kind = "linear", bounds_error = True, axis = 0)(hFreq))

print("Combine Voice Source and Voice Tract...")
for iHop, f0 in enumerate(f0List):
    if(f0 <= 0):
        continue
    need = hFreqList[iHop] > 0
    hFreq = hFreqList[iHop][need]
    hVsPhase = hVsPhaseList[iHop][need]
    hVtAmp = hVtAmpList[iHop][need]

    hVsAmp = np.abs(lfmodel.calcSpectrum(np.asarray(hFreq), 1 / f0, 1.0, *lfmodel.calcParameterFromRd(rdList[iHop])))
    hVsAmp /= hVsAmp[0]
    hAmpList[iHop][need] = hVsAmp * hVtAmp

    hVtPhase = calcSinusoidMinphase(hFreq, hVtAmp)
    hPhaseList[iHop][need] = wrap(hVsPhase + hVtPhase)

print("Phase Propagate for Voice Source...")
tList = np.arange(nHop) * hopSize / sr
hPhaseList = propagatePhase(hFreqList, hPhaseList, hopSize, sr, False)

print("HNM Synthing...")
synProc = hnm.Synther(sr)
synthed = synProc(hFreqList, hAmpList, hPhaseList, sinusoidEnergyList, noiseEnvList, noiseEnergyList)
print("Finished.")