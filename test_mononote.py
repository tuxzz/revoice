import numpy as np
import pylab as pl
from revoice import *
from revoice.common import *
import matplotlib.patches as patches

w, sr = loadWav("voices/tuku_ra.wav")
energyList = energy.Analyzer(sr)(w)

print("PYin...")
pyinAnalyzer = pyin.Analyzer(sr)
obsProbList = pyinAnalyzer(w)

print("Monopitch...")
monopitchAnalyzer = monopitch.Analyzer(*monopitch.parameterFromPYin(pyinAnalyzer))
f0List = monopitchAnalyzer(obsProbList)
silentList = energyList < 1e-8
f0List[silentList] = -np.abs(f0List[silentList])

print("Mononote...")
mononoteAnalyzer = mononote.Analyzer(*mononote.parameterFromPYin(pyinAnalyzer))
noteList = mononoteAnalyzer(energyList, obsProbList)

fig = pl.figure()
ax = pl.subplot(111)
pl.plot(np.arange(w.shape[0]) / sr, w * 100)
tList = np.arange(f0List.shape[0]) * pyinAnalyzer.hopSize / sr
pl.plot(tList, f0List)
for iNote, note in enumerate(noteList):
    freq = pitchToFreq(note["pitch"])
    begin = note["begin"] * pyinAnalyzer.hopSize / sr
    end = note["end"] * pyinAnalyzer.hopSize / sr
    ax.add_patch(patches.Rectangle((begin, freq - 5.0), end - begin, 10.0, alpha = 0.39))
pl.show()
