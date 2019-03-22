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

w = sp.resample_poly(w, int(round(w.shape[0] / sr * 12000.0)), w.shape[0]).astype(np.float32)
sinusoid = sp.resample_poly(sinusoid, int(round(sinusoid.shape[0] / sr * 12000.0)), sinusoid.shape[0]).astype(np.float32)
lpcSr = 12000.0

print("LPC Analyzing...")
lpcProc = lpc.Analyzer(lpcSr, order = order)
coeffList, xmsList = lpcProc(sinusoid, f0List)

lpcSpectrum = np.zeros((nHop, 2049))
FList, bwList = np.zeros((nHop, int(np.ceil(order / 2))), dtype=np.float32), np.zeros((nHop, int(np.ceil(order / 2))), dtype=np.float32)
for iHop, f0 in enumerate(f0List):
    lpcSpectrum[iHop] = lpc.calcMagnitudeFromLPC(coeffList[iHop], xmsList[iHop], fftSize, lpcSr, deEmphasisFreq = 50.0)
    F, bw = lpc.calcFormantFromLPC(coeffList[iHop], lpcSr)
    need = np.logical_and(F > 50.0, F < lpcSr * 0.5)
    F, bw = F[need], bw[need]
    FList[iHop, :F.shape[0]], bwList[iHop, :F.shape[0]] = F, bw

FList = FList[:, :nMaxFormant]
bwList = bwList[:, :nMaxFormant]

print("Formant Tracking...")
formantTracker = formanttracker.Analyzer(nMaxFormant, lpcSr)
trackedFList, trackedBwList = formantTracker(hFreqList, hAmpList, FList, bwList)
assert trackedFList.dtype == trackedBwList.dtype == np.float32
for iFormant in range(nMaxFormant):
    iHop = 0
    splittedF0List = splitArray(f0List)
    for splittedF0 in splittedF0List:
        iEndHop = iHop + splittedF0.shape[0]
        if(splittedF0[0] > 0):
            trackedFList[iHop:iEndHop, iFormant] = applySmoothingFilter(trackedFList[iHop:iEndHop, iFormant], 9)
        else:
            trackedFList[iHop:iEndHop, iFormant] = 0
        iHop = iEndHop

#need = f0List <= 0
#FList[need] = 0
#trackedFList[need] = 0
need = FList <= 0
FList[need] = np.nan
need = trackedFList <= 0
trackedFList[need] = np.nan

lpcSpectrum = np.log(np.clip(lpcSpectrum, 1e-6, np.inf))
tList = np.arange(f0List.shape[0]) * pyinAnalyzer.hopSize / sr
pl.imshow(lpcSpectrum.T, interpolation = 'bicubic', aspect = 'auto', origin = 'lower', extent = [tList[0], tList[-1], 0, lpcSr / 2])
pl.plot(tList, FList, 'o')
pl.plot(tList, trackedFList)
pl.show()
