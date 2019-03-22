import numpy as np
import pylab as pl
from revoice import *
from revoice.common import *

w, sr = loadWav("voices/yuri_orig.wav")
energyList = energy.Analyzer(sr)(w)

print("Direct PYin...")
pyinAnalyzer = pyin.Analyzer(sr, prefilter = False)
obsProbList = pyinAnalyzer(w)
print("Direct Monopitch...")
monopitchAnalyzer = monopitch.Analyzer(*monopitch.parameterFromPYin(pyinAnalyzer))
f0List = monopitchAnalyzer(obsProbList)
assert f0List.dtype == np.float32

print("Prefiltered PYin...")
pyinAnalyzer = pyin.Analyzer(sr)
obsProbList = pyinAnalyzer(w)
print("Prefiltered Monopitch...")
monopitchAnalyzer = monopitch.Analyzer(*monopitch.parameterFromPYin(pyinAnalyzer))
f0List_pf = monopitchAnalyzer(obsProbList)
assert f0List_pf.dtype == np.float32

silentList = energyList < 1e-8
f0List[silentList] = -np.abs(f0List[silentList])
f0List_pf[silentList] = -np.abs(f0List_pf[silentList])

pl.plot(np.arange(w.shape[0]) / sr, w * 100)
tList = np.arange(f0List.shape[0]) * pyinAnalyzer.hopSize / sr
pl.plot(tList, f0List, label = "Direct")
pl.plot(tList, f0List_pf, label = "Prefiltered")
pl.legend()
pl.show()
