import numpy as np
import pylab as pl
from revoice import *
from revoice.common import *

w, sr = loadWav("voices/yuri_orig.wav")
energyList = energy.Analyzer(sr)(w)

print("F0 Estimation...")
pyinAnalyzer = pyin.Analyzer(sr)
obsProbList = pyinAnalyzer(w)
monopitchAnalyzer = monopitch.Analyzer(*monopitch.parameterFromPYin(pyinAnalyzer))
f0List = monopitchAnalyzer(obsProbList)
silentList = energyList < 1e-8
f0List[silentList] = -np.abs(f0List[silentList])

print("F0 Refinement...")
f0RefineProcessor = refinef0_stft.Processor(sr)
f0List_rf = f0RefineProcessor(w, f0List)
assert f0List_rf.dtype == np.float32

pl.plot(np.arange(w.shape[0]) / sr, w * 100)
tList = np.arange(f0List.shape[0]) * pyinAnalyzer.hopSize / sr
pl.plot(tList, f0List, label = "Estimation")
pl.plot(tList, f0List_rf, label = "Refined")
pl.legend()
pl.show()
