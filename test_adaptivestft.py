import numpy as np
import pylab as pl
from revoice import *
from revoice.common import *

w, sr = loadWav("voices/yuri_orig.wav")
energyList = energy.Analyzer(sr)(w)

print("PYin...")
pyinAnalyzer = pyin.Analyzer(sr)
obsProbList = pyinAnalyzer(w)

print("Monopitch...")
monopitchAnalyzer = monopitch.Analyzer(*monopitch.parameterFromPYin(pyinAnalyzer))
f0List = monopitchAnalyzer(obsProbList)
silentList = energyList < 1e-8
f0List[silentList] = -np.abs(f0List[silentList])

print("Adaptive STFT...")
stftAnalyzer = adaptivestft.Analyzer(sr)
fSigList = stftAnalyzer(w, f0List)

magnList = np.abs(fSigList)
tx = (np.arange(f0List.shape[0]) + 0.5) * pyinAnalyzer.hopSize / sr
pl.imshow(np.log(np.clip(magnList, 1e-8, np.inf)).T, interpolation = 'bicubic', aspect = 'auto', origin = 'lower', extent = [tx[0], tx[-1], 0, sr / 2])
pl.plot(tx, f0List, label = 'original f0')
pl.legend()
pl.show()
