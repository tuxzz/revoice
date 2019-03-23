import numpy as np
import pylab as pl
from revoice import *
from revoice.common import *

#w, sr = loadWav("voices/yuri_orig.wav")
w, sr = loadWav("voices/square110880.wav")
energyList = energy.Analyzer(sr)(w)

print("F0 Detecting...")
hubbleAnalyzer = hubble.Analyzer(sr)
f0List = hubbleAnalyzer(w)

print("HNM Analyzing...")
hnmAnalyzer = hnm.Analyzer(sr, harmonicAnalysisMethod = "qhm")
hFreqList, hAmpList, hPhaseList, sinusoidEnergyList, noiseEnvList, noiseEnergyList = hnmAnalyzer(w, f0List)
assert hFreqList.dtype == hAmpList.dtype == hPhaseList.dtype == sinusoidEnergyList.dtype == noiseEnvList.dtype == noiseEnergyList.dtype == np.float32

print("Sinusoid Synthing...")
synProc = hnm.Synther(sr)
sinusoid = synProc(hFreqList, hAmpList, hPhaseList, sinusoidEnergyList, None, None, enableNoise = False)
srer = calcSRER(w, sinusoid)
assert sinusoid.dtype == np.float32

print("HNM Synthing...")
synProc = hnm.Synther(sr)
synthed = synProc(hFreqList, hAmpList, hPhaseList, sinusoidEnergyList, noiseEnvList, noiseEnergyList)
assert synthed.dtype == np.float32

print("Average SRER = %lf" % srer)
tList = np.arange(f0List.shape[0]) * hubbleAnalyzer.hopSize / sr
hFreqList[hFreqList <= 0.0] = np.nan
pl.figure()
pl.plot(tList, hFreqList)
pl.figure()
pl.plot(tList, sinusoidEnergyList)
pl.plot(tList, noiseEnergyList)
pl.figure()
pl.plot(np.arange(w.shape[0]) / sr, synthed)
pl.show()
