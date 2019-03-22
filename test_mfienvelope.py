import numpy as np
import pylab as pl
from revoice import *
from revoice.common import *
import scipy.fftpack as sfft

w, sr = loadWav("voices/chihaya_01.wav")
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

print("MFI Envelope...")
envAnalyzer = mfienvelope.Analyzer(sr)
envList = envAnalyzer(w, f0List)
assert envList.dtype == np.float32

print("Plotting...")
t = w.shape[0] / sr
tx = (np.arange(f0List.shape[0]) + 0.5) * pyinAnalyzer.hopSize / sr
pl.imshow(envList.T, interpolation = 'bicubic', aspect = 'auto', origin = 'lower', extent = [tx[0], tx[-1], 0, sr / 2])
pl.plot(tx, f0List, label = 'original f0')
pl.figure()
pl.plot(np.log(np.clip(magnList[100], 1e-6, np.inf)))
pl.plot(envList[100])
pl.show()
