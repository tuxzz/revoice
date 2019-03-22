import numpy as np
import pylab as pl
from revoice import *
from revoice.common import *

w, sr = loadWav("voices/yuri_orig.wav")

b, a = dc_iir_filter(50.0 / sr / 2)
w = sp.filtfilt(b, a, w)

nBark = 128
barkStart = freqToBark(40.0)
barkStop = freqToBark(1000.0)
barkFreqList = barkToFreq(np.linspace(barkStart, barkStop, nBark, dtype=np.float32))
barkFFTSize = roundUpToPowerOf2(sr / np.min(np.diff(barkFreqList)))

x = w
hopSize = sr * 0.0025
nX = len(w)
nHop = getNFrame(nX, hopSize)

fftSize = int(np.ceil(sr / 40.0 * 4))#roundUpToPowerOf2(sr * 0.025)

snrList = np.zeros((nHop, nBark), dtype=np.float32)
for iFreq, freq in enumerate(barkFreqList):
  #print(yang.calcYangSNR(frame, freq / sr, freq / sr))
  snrList[:, iFreq] = yang.calcYangSNR(w, freq / sr, freq / sr)[(np.arange(0, nHop, dtype=np.float64) * hopSize).round().astype(np.int32)]

pl.imshow(snrList.T, cmap='plasma', interpolation='nearest', aspect='auto', origin='lower')
pl.show()
