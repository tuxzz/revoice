import numpy as np
import pylab as pl
from revoice import *
from revoice.common import *
import os, pickle

path = "./voices/yuri_orig.wav"

nMaxFormant = 4
lpcSr = 12000.0
formantFreqFac = np.array([0.75, 0.875, 0.9, 0.925])
formantBwFac = np.array([1.0, 1.0, 1.0, 1.0])
formantAmpFac = np.array([1.0, 1.0, 1.0, 1.0])
vtResidualFreqFac = 0.95
vtResidualAmpFac = 1.0
timeFac = 1.0
pitchFac = 0.75
fixedPitch = None
rdFac = 1.1
hnmAnalysisMethod = "get"
cachePath = "%s.%s.%s" % (path, hnmAnalysisMethod, "cache.pickle")

if(os.path.isfile(cachePath)):
    print("Load cache...")
    with open(cachePath, "rb") as f:
        w, sr, energyList, hFreqList, hAmpList, hPhaseList, sinusoidEnergyList, noiseEnvList, noiseEnergyList, hopSize, nHop, FList, bwList, ampList, envList, sinusoid, hVtAmpList, hVsPhaseList, f0List, rdList = pickle.load(f)
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

    print("Voice Tract Analyzing...")
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

    rsSinusoid = sp.resample_poly(sinusoid, int(np.round(sinusoid.shape[0] / sr * lpcSr)), sinusoid.shape[0])

    print("LPC Analyzing...")
    lpcOrder = nMaxFormant * 2 + 1
    lpcProc = lpc.Analyzer(lpcSr, order = lpcOrder)
    coeffList, xmsList = lpcProc(rsSinusoid, f0List)

    FList, bwList = np.zeros((nHop, int(np.ceil(lpcOrder * 0.5)))), np.zeros((nHop, int(np.ceil(lpcOrder * 0.5))))
    for iHop, f0 in enumerate(f0List):
        F, bw = lpc.calcFormantFromLPC(coeffList[iHop], lpcSr)
        need = np.logical_and(F > 50.0, F < sr * 0.5)
        F, bw = F[need], bw[need]
        FList[iHop, :F.shape[0]], bwList[iHop, :F.shape[0]] = F, bw

    FList = FList[:, :nMaxFormant]
    bwList = bwList[:, :nMaxFormant]

    print("Formant Tracking...")
    formantTracker = formanttracker.Analyzer(nMaxFormant, lpcSr)
    FList, bwList = formantTracker(hFreqList, hVtAmpList, FList, bwList)
    for iFormant in range(nMaxFormant):
        iHop = 0
        splittedF0List = splitArray(f0List)
        for splittedF0 in splittedF0List:
            iEndHop = iHop + splittedF0.shape[0]
            if(splittedF0[0] > 0):
                FList[iHop:iEndHop, iFormant] = applySmoothingFilter(FList[iHop:iEndHop, iFormant], 9)
            else:
                FList[iHop:iEndHop, iFormant] = 0
            iHop = iEndHop

    print("Formant Refinement...")
    formantRefineProcessor = refineformant.Processor(lpcSr)
    FList, bwList, ampList = formantRefineProcessor(hFreqList, hVtAmpList, FList, bwList)
    with open(cachePath, "wb") as f:
        pickle.dump((w, sr, energyList, hFreqList, hAmpList, hPhaseList, sinusoidEnergyList, noiseEnvList, noiseEnergyList, hopSize, nHop, FList, bwList, ampList, envList, sinusoid, hVtAmpList, hVsPhaseList, f0List, rdList), f)

print("Formant Refinement Filter...")
for iFormant in range(nMaxFormant):
    iHop = 0
    splittedF0List = splitArray(f0List)
    for splittedF0 in splittedF0List:
        iEndHop = iHop + splittedF0.shape[0]
        if(splittedF0[0] > 0):
            bwList[iHop:iEndHop, iFormant] = applySmoothingFilter(bwList[iHop:iEndHop, iFormant], 9)
            ampList[iHop:iEndHop, iFormant] = applySmoothingFilter(ampList[iHop:iEndHop, iFormant], 9)
        else:
            ampList[iHop:iEndHop, iFormant] = 0
        iHop = iEndHop

print("Splitting Voice Tract...")
vtResidualEnvList = np.zeros(envList.shape)
vtResidualNoiseEnvList = np.zeros(noiseEnvList.shape)
f = rfftFreq((envList.shape[1] - 1) * 2, sr)
for iHop, f0 in enumerate(f0List):
    if(f0 <= 0):
        vtResidualNoiseEnvList[iHop] = noiseEnvList[iHop]
        continue
    formantEnv = np.log(calcKlattFilterBankResponseMagnitude(f, FList[iHop], bwList[iHop], ampList[iHop], sr))
    vtResidualEnvList[iHop] = envList[iHop] - formantEnv
    vtResidualNoiseEnvList[iHop] = noiseEnvList[iHop] - formantEnv
del f, envList, noiseEnvList

print("Inverse Phase Propagate for Voice Source...")
tList = np.arange(nHop) * hopSize / sr
hVsPhaseList = propagatePhase(hFreqList, hVsPhaseList, hopSize, sr, True)
hVsPhaseList = np.unwrap(hVsPhaseList, axis = 0)

print("Formant Shifting...")
FList *= formantFreqFac
bwList *= formantBwFac
ampList *= formantAmpFac

print("Voice Tract Residual Amp Shifting...")
vtResidualEnvList *= vtResidualAmpFac
vtResidualNoiseEnvList *= vtResidualAmpFac

print("Voice Tract Residual Freq Shifting...")
vtResidualEnvList += np.log(vtResidualAmpFac)
vtResidualNoiseEnvList += np.log(vtResidualAmpFac)

f = rfftFreq((vtResidualEnvList.shape[1] - 1) * 2, sr)
vtResidualEnvList = ipl.interp1d(f * vtResidualFreqFac, vtResidualEnvList, kind = "linear", axis = 1, bounds_error = False, fill_value = vtResidualEnvList[:, -1])(f)
vtResidualNoiseEnvList = ipl.interp1d(f * vtResidualFreqFac, vtResidualNoiseEnvList, kind = "linear", axis = 1, bounds_error = False, fill_value = vtResidualEnvList[:, -1])(f)

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
f0List = hFreqList[:, 0]

print("Time Shifting...")
splittedF0List = splitArray(f0List)
splittedFList = splitArray(FList[:, 0])
nNewHop = int(nHop * timeFac)

# These are everything we need
newRdList = np.zeros(nNewHop)
newVtResidualEnvList = np.zeros((nNewHop, vtResidualEnvList.shape[1]))
newVtResidualNoiseEnvList = np.zeros((nNewHop, vtResidualNoiseEnvList.shape[1]))
newF0List = np.zeros(nNewHop)
newHVsPhaseList = np.zeros((nNewHop, hVsPhaseList.shape[1]))
newSinusoidEnergyList = np.zeros(nNewHop)
newNoiseEnergyList = ipl.interp1d(np.arange(nHop), noiseEnergyList, kind = "linear", bounds_error = True, axis = 0)(np.linspace(0, nHop - 1, nNewHop))
newFList = np.zeros((nNewHop, nMaxFormant))
newBwList = np.zeros((nNewHop, nMaxFormant))
newAmpList = np.zeros((nNewHop, nMaxFormant))

iHop = 0
for segment in splittedF0List:
    segmentSize = len(segment)

    hopEnd = iHop + segmentSize
    iNewHop = int(iHop * timeFac)
    newHopEnd = int(hopEnd * timeFac)

    hopMapSource = np.arange(iHop, hopEnd)
    hopMap = np.linspace(iHop, hopEnd - 1, newHopEnd - iNewHop)
    newVtResidualNoiseEnvList[iNewHop:newHopEnd] = ipl.interp1d(hopMapSource, vtResidualNoiseEnvList[iHop:hopEnd], kind = "linear", bounds_error = True, axis = 0)(hopMap)
    
    if(segment[0] <= 0):
        iHop += segmentSize
        continue

    newRdList[iNewHop:newHopEnd] = ipl.interp1d(hopMapSource, rdList[iHop:hopEnd], kind = "linear", bounds_error = True, axis = 0)(hopMap)
    newVtResidualEnvList[iNewHop:newHopEnd] = ipl.interp1d(hopMapSource, vtResidualEnvList[iHop:hopEnd], kind = "linear", bounds_error = True, axis = 0)(hopMap)
    newFList[iNewHop:newHopEnd] = ipl.interp1d(hopMapSource, FList[iHop:hopEnd], kind = "linear", bounds_error = True, axis = 0)(hopMap)
    newBwList[iNewHop:newHopEnd] = ipl.interp1d(hopMapSource, bwList[iHop:hopEnd], kind = "linear", bounds_error = True, axis = 0)(hopMap)
    newAmpList[iNewHop:newHopEnd] = ipl.interp1d(hopMapSource, ampList[iHop:hopEnd], kind = "linear", bounds_error = True, axis = 0)(hopMap)
    newF0List[iNewHop:newHopEnd] = ipl.interp1d(hopMapSource, f0List[iHop:hopEnd], kind = "linear", bounds_error = True, axis = 0)(hopMap)
    newHVsPhaseList[iNewHop:newHopEnd] = ipl.interp1d(hopMapSource, hVsPhaseList[iHop:hopEnd], kind = "linear", bounds_error = True, axis = 0)(hopMap)
    newSinusoidEnergyList[iNewHop:newHopEnd] = ipl.interp1d(hopMapSource, sinusoidEnergyList[iHop:hopEnd], kind = "linear", bounds_error = True, axis = 0)(hopMap)

    iHop += segmentSize

# regenerate harmonic frequency
newHFreqList = newF0List.reshape(nNewHop, 1) * np.arange(1, hFreqList.shape[1] + 1)
newHFreqList[np.logical_or(newHFreqList < 0.0, newHFreqList >= sr / 2)] = 0.0

rdList = newRdList
vtResidualEnvList = newVtResidualEnvList
hFreqList = newHFreqList
hVsPhaseList = newHVsPhaseList
sinusoidEnergyList = newSinusoidEnergyList
vtResidualNoiseEnvList = newVtResidualNoiseEnvList
noiseEnergyList = newNoiseEnergyList
FList = newFList
bwList = newBwList
ampList = newAmpList
nHop = nNewHop
f0List = hFreqList[:, 0]

print("Recovery Envelope from Formant and Residual...")
f = rfftFreq((vtResidualEnvList.shape[1] - 1) * 2, sr)
envList = np.zeros(vtResidualEnvList.shape)
noiseEnvList = np.zeros(vtResidualNoiseEnvList.shape)
for iHop, f0 in enumerate(f0List):
    if(f0 <= 0):
        noiseEnvList[iHop] = vtResidualNoiseEnvList[iHop]
        continue
    formantEnv = np.log(calcKlattFilterBankResponseMagnitude(f, FList[iHop], bwList[iHop], ampList[iHop], sr))
    envList[iHop] = vtResidualEnvList[iHop] + formantEnv
    noiseEnvList[iHop] = vtResidualNoiseEnvList[iHop] + formantEnv

print("Recovery Voice Tract Amplitude from Envelope...")
hVtAmpList = np.zeros(hFreqList.shape)
hAmpList = np.zeros(hFreqList.shape)
hPhaseList = np.zeros(hFreqList.shape)
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