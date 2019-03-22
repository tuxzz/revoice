import numpy as np
import pylab as pl
from revoice import *
from revoice.common import *

w, sr = loadWav("voices/yuri_orig.wav")

print("Direct...")
yinAnalyzer = yin.Analyzer(sr, prefilter = False)
f0List = yinAnalyzer(w)
assert f0List.dtype == np.float32

print("Prefiltered...")
yinAnalyzer = yin.Analyzer(sr)
f0List_pf = yinAnalyzer(w)
assert f0List_pf.dtype == np.float32

pl.plot(np.arange(w.shape[0]) / sr, w * 100)
tList = np.arange(f0List.shape[0]) * yinAnalyzer.hopSize / sr
pl.plot(tList, f0List, label = "Direct")
pl.plot(tList, f0List_pf, label = "Prefiltered")
pl.legend()
pl.show()
