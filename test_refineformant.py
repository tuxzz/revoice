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

FList, bwList = np.zeros((nHop, int(np.ceil(order * 0.5)))), np.zeros((nHop, int(np.ceil(order * 0.5))))
for iHop, f0 in enumerate(f0List):
    F, bw = lpc.calcFormantFromLPC(coeffList[iHop], sr)
    need = np.logical_and(F > 50.0, F < sr * 0.5)
    F, bw = F[need], bw[need]
    FList[iHop, :F.shape[0]], bwList[iHop, :F.shape[0]] = F, bw

FList = FList[:, :nMaxFormant]
bwList = bwList[:, :nMaxFormant]

print("Formant Tracking...")
formantTracker = formanttracker.Analyzer(nMaxFormant, sr)
FList, bwList = formantTracker(hFreqList, hAmpList, FList, bwList)
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
formantRefineProcessor = refineformant.Processor(sr)
FList, bwList, ampList = formantRefineProcessor(hFreqList, hAmpList, FList, bwList)

print("Adaptive STFT...")
stftAnalyzer = adaptivestft.Analyzer(sr)
fSigList = stftAnalyzer(sinusoid, f0List)
magnList = np.abs(fSigList)

'''
#need = f0List <= 0
#FList[need] = 0
#trackedFList[need] = 0
need = FList <= 0
FList[need] = np.nan
need = FList <= 0
FList[need] = np.nan
'''

need = f0List > 0
iHop = 150
pl.plot(hFreqList[need][iHop], np.log(hAmpList[need][iHop]))
pl.plot(rfftFreq(stftAnalyzer.fftSize, sr), np.log(calcKlattFilterBankResponseMagnitude(rfftFreq(stftAnalyzer.fftSize, sr), FList[need][iHop], bwList[need][iHop], ampList[need][iHop], sr)))
pl.show()