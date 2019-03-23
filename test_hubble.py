import numpy as np
import pylab as pl
from revoice import *
from revoice.common import *

b, a = dc_iir_filter(50.0 / hubble.workSr / 2)

w, sr = loadWav("voices/yuki_01.wav")
pw = sp.resample_poly(w, hubble.workSr, sr).astype(np.float32)
pw = sp.filtfilt(b, a, pw).astype(np.float32)

energyAnalyzer = energy.Analyzer(sr)
hopSize = energyAnalyzer.hopSize
energyList = energyAnalyzer(w)
nHop = energyList.size

print("Hubble generate...")
f0ProbMapList = hubble.generateMap(pw, hopSize / sr * hubble.workSr)

print("Hubble track...")
path = hubble.Tracker()(f0ProbMapList)
pathFull = hubble.F0Tracker()(f0ProbMapList)

f0List = hubble.binToFreq(path)
f0ListFull = hubble.binToFreq(pathFull)

print("Hubble refine...")
refinedF0List = hubble.refine(pw, f0List, hopSize / sr * hubble.workSr)
refinedF0ListFull = hubble.refine(pw, f0ListFull, hopSize / sr * hubble.workSr)

silentList = energyList < 1e-8
refinedF0List[silentList] = np.nan
#refinedF0ListFull[silentList] = np.nan

print("Yang...")
snrList = np.zeros((nHop, 1100), dtype=np.float32)
for iFreq in range(66, 1100):
  freq = iFreq
  rf = freq / hubble.workSr
  ww = yang.createYangSNRParameter(rf)
  snrList[:, iFreq] = yang.calcYangSNR(pw, rf, ww)[(np.arange(0, nHop, dtype=np.float64) * hopSize / sr * hubble.workSr).round().astype(np.int32)]

pl.figure()
pl.plot(np.arange(w.shape[0]) / sr, w * 100)
tList = np.arange(f0List.shape[0]) * hopSize / sr
pl.plot(tList, f0List, label="VUV")
pl.plot(tList, f0ListFull, label="Full")
pl.plot(tList, refinedF0List, label="Refined VUV")
pl.plot(tList, refinedF0ListFull, label="Refined Full")
pl.legend()
pl.figure()
pl.title("SNR")
pl.imshow(snrList.T, cmap='plasma', interpolation='nearest', aspect='auto', origin='lower')
pl.plot(f0List, label="VUV")
pl.plot(refinedF0List, label="Refined VUV")
pl.show()
