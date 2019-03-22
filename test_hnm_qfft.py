import numpy as np
import pylab as pl
from revoice import *
from revoice.common import *

#w, sr = loadWav("voices/yuri_orig.wav")
w, sr = loadWav("voices/square110880.wav")
energyList = energy.Analyzer(sr)(w)

print("F0 Estimation...")
pyinAnalyzer = pyin.Analyzer(sr)
obsProbList = pyinAnalyzer(w)
monopitchAnalyzer = monopitch.Analyzer(*monopitch.parameterFromPYin(pyinAnalyzer))
f0List = monopitchAnalyzer(obsProbList)
silentList = energyList < 1e-8
f0List[silentList] = -np.abs(f0List[silentList])
'''
# fix vuv flag before we have better vuv detector
# fix for voices/yuri_orig.wav
f0List[int(0.19 * sr / pyinAnalyzer.hopSize):int(0.354 * sr / pyinAnalyzer.hopSize) + 1] = np.abs(f0List[int(0.19 * sr / pyinAnalyzer.hopSize):int(0.354 * sr / pyinAnalyzer.hopSize) + 1])
f0List[int(2.669 * sr / pyinAnalyzer.hopSize):int(2.689 * sr / pyinAnalyzer.hopSize) + 1] = np.abs(f0List[int(2.669 * sr / pyinAnalyzer.hopSize):int(2.689 * sr / pyinAnalyzer.hopSize) + 1])
f0List[int(2.814 * sr / pyinAnalyzer.hopSize):int(2.86 * sr / pyinAnalyzer.hopSize) + 1] = np.abs(f0List[int(2.814 * sr / pyinAnalyzer.hopSize):int(2.86 * sr / pyinAnalyzer.hopSize) + 1])
'''
print("F0 Refinement...")
f0RefineProcessor = refinef0_stft.Processor(sr)
f0List = f0RefineProcessor(w, f0List)

print("HNM Analyzing...")
hnmAnalyzer = hnm.Analyzer(sr, harmonicAnalysisMethod = "qfft")
hFreqList, hAmpList, hPhaseList, sinusoidEnergyList, noiseEnvList, noiseEnergyList = hnmAnalyzer(w, f0List)
assert hFreqList.dtype == hAmpList.dtype == hPhaseList.dtype == sinusoidEnergyList.dtype == noiseEnvList.dtype == noiseEnergyList.dtype == np.float32

print("Sinusoid Synthing...")
synProc = hnm.Synther(sr)
sinusoid = synProc(hFreqList, hAmpList, hPhaseList, sinusoidEnergyList, None, None, enableNoise = False)
srer = calcSRER(w[:min(w.shape[0], sinusoid.shape[0])], sinusoid[:min(w.shape[0], sinusoid.shape[0])])
assert sinusoid.dtype == np.float32

print("HNM Synthing...")
synProc = hnm.Synther(sr)
synthed = synProc(hFreqList, hAmpList, hPhaseList, sinusoidEnergyList, noiseEnvList, noiseEnergyList)
assert synthed.dtype == np.float32

print("Average SRER = %lf" % srer)
tList = np.arange(f0List.shape[0]) * pyinAnalyzer.hopSize / sr
hFreqList[hFreqList <= 0.0] = np.nan
pl.figure()
pl.plot(tList, hFreqList)
pl.figure()
pl.plot(tList, sinusoidEnergyList)
pl.plot(tList, noiseEnergyList)
pl.figure()
pl.plot(np.arange(synthed.shape[0]) / sr, synthed)
pl.show()