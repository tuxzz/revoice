import numpy as np
import pylab as pl
from revoice import *
from revoice.common import *

w, sr = loadWav("voices/yuri_orig.wav")

print("Direct...")
pyinAnalyzer = pyin.Analyzer(sr, prefilter = False)
obsProbList = pyinAnalyzer(w)
f0List = pyin.extractF0(obsProbList)

print("Prefiltered...")
pyinAnalyzer = pyin.Analyzer(sr)
obsProbList = pyinAnalyzer(w)
f0List_pf = pyin.extractF0(obsProbList)

pl.plot(np.arange(w.shape[0]) / sr, w * 100)
tList = np.arange(f0List.shape[0]) * pyinAnalyzer.hopSize / sr
pl.plot(tList, f0List, label = "Direct")
pl.plot(tList, f0List_pf, label = "Prefiltered")
pl.legend()
pl.show()
