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
magnList = np.abs(fSigList, dtype=np.float32)
assert fSigList.dtype == np.complex64
assert magnList.dtype == np.float32

print("Bark Scale...")
nFilter = int(np.ceil(freqToBark(sr / 2) / 1)) + 1
barkFilterBank = adaptivestft.calcBarkScaledSpectrumFilterBank(nFilter, 0.0, 1.0, stftAnalyzer.fftSize, sr)
barkSpectrum = adaptivestft.applyFilterBank(magnList[100], barkFilterBank)
barkSpectrogram = adaptivestft.applyFilterBank(magnList, barkFilterBank)
assert barkSpectrum.dtype == np.float32
assert barkSpectrogram.dtype == np.float32

linearBarkSpectrum = adaptivestft.calcLinearMagnFromBark(barkSpectrum, nFilter, 0.0, 1.0, stftAnalyzer.fftSize, sr)
linearBarkSpectrogram = adaptivestft.calcLinearMagnFromBark(barkSpectrogram, nFilter, 0.0, 1.0, stftAnalyzer.fftSize, sr)
assert linearBarkSpectrum.dtype == np.float32
assert linearBarkSpectrogram.dtype == np.float32

tx = (np.arange(f0List.shape[0]) + 0.5) * pyinAnalyzer.hopSize / sr
pl.figure()
pl.imshow(np.log(np.clip(magnList, 1e-8, np.inf)).T, interpolation = 'bicubic', aspect = 'auto', origin = 'lower', extent = [tx[0], tx[-1], 0, sr / 2])
pl.plot(tx, f0List, label = 'original f0')
pl.legend()

pl.figure()
pl.plot(np.log(magnList[100] + eps), label = 'linear spectrum')
pl.plot(np.log(linearBarkSpectrum + eps), label = 'bark spectrum')
pl.legend()

pl.show()
