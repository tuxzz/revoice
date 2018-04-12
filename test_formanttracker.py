import numpy as np
import pylab as pl
from revoice import *
from revoice.common import *
import scipy.fftpack as sfft

w, sr = loadWav("voices/yuri_orig.wav")
w = applyPreEmphasisFilter(w, 50.0, sr)

nMaxFormant = 4
order = nMaxFormant * 2 + 1
fftSize = 4096

print("PYin...")
pyinAnalyzer = pyin.Analyzer(sr)
obsProbList = pyinAnalyzer(w)
energyList = energy.Analyzer(sr)(w)
nHop = energyList.shape[0]

print("Monopitch...")
monopitchAnalyzer = monopitch.Analyzer(*monopitch.parameterFromPYin(pyinAnalyzer))
f0List = monopitchAnalyzer(obsProbList)
silentList = energyList < 1e-8
f0List[silentList] = -np.abs(f0List[silentList])

print("F0 Refinement...")
f0RefineProcessor = refinef0_stft.Processor(sr)
f0List = f0RefineProcessor(w, f0List)

print("HNM Analyzing...")
hnmAnalyzer = hnm.Analyzer(sr, harmonicAnalysisMethod = "get")
hFreqList, hAmpList, hPhaseList, sinusoidEnergyList, noiseEnvList, noiseEnergyList = hnmAnalyzer(w, f0List)

print("Sinusoid Synthing...")
synProc = hnm.Synther(sr)
sinusoid = synProc(hFreqList, hAmpList, hPhaseList, sinusoidEnergyList, None, None, enableNoise = False)

w = sp.resample_poly(w, int(np.round(w.shape[0] / sr * 12000.0)), w.shape[0])
sinusoid = sp.resample_poly(sinusoid, int(np.round(sinusoid.shape[0] / sr * 12000.0)), sinusoid.shape[0])
sr = 12000.0

print("LPC Analyzing...")
lpcProc = lpc.Analyzer(sr, order = order)
coeffList, xmsList = lpcProc(sinusoid, f0List)

lpcSpectrum = np.zeros((nHop, 2049))
FList, bwList = np.zeros((nHop, int(np.ceil(order * 0.5)))), np.zeros((nHop, int(np.ceil(order * 0.5))))
for iHop, f0 in enumerate(f0List):
    lpcSpectrum[iHop] = lpc.calcMagnitudeFromLPC(coeffList[iHop], xmsList[iHop], fftSize, sr, deEmphasisFreq = 50.0)
    F, bw = lpc.calcFormantFromLPC(coeffList[iHop], sr)
    need = np.logical_and(F > 50.0, F < sr * 0.5)
    F, bw = F[need], bw[need]
    FList[iHop, :F.shape[0]], bwList[iHop, :F.shape[0]] = F, bw

FList = FList[:, :nMaxFormant]
bwList = bwList[:, :nMaxFormant]

print("Formant Tracking...")
formantTracker = formanttracker.Analyzer(nMaxFormant, sr)
trackedFList, trackedBwList = formantTracker(hFreqList, hAmpList, FList, bwList)

for iFormant in range(nMaxFormant):
    n = 0
    splittedFList = splitArray(trackedFList[:, iFormant])
    for splittedF in splittedFList:
        splittedF = applySmoothingFilter(splittedF, 13)
        trackedFList[n:n + splittedF.shape[0], iFormant] = splittedF
        n += splittedF.shape[0]

#need = f0List <= 0
#FList[need] = 0
#trackedFList[need] = 0
need = FList <= 0
FList[need] = np.nan
need = trackedFList <= 0
trackedFList[need] = np.nan

lpcSpectrum = np.log(np.clip(lpcSpectrum, 1e-6, np.inf))
tList = np.arange(f0List.shape[0]) * pyinAnalyzer.hopSize / sr
pl.imshow(lpcSpectrum.T, interpolation = 'bicubic', aspect = 'auto', origin = 'lower', extent = [tList[0], tList[-1], 0, sr / 2])
pl.plot(tList, FList, 'o')
pl.plot(tList, trackedFList)
pl.show()
