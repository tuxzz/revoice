import numpy as np
import pylab as pl
from revoice import *
from revoice.common import *

w, sr = loadWav("voices/yuri_orig.wav")

b, a = dc_iir_filter(50.0 / sr / 2)
w = sp.filtfilt(b, a, w).astype(np.float32)

nBark = 64
barkStart = freqToBark(40.0)
barkStop = freqToBark(1000.0)
barkFreqList = barkToFreq(np.linspace(barkStart, barkStop, nBark, dtype=np.float32))
barkFFTSize = roundUpToPowerOf2(sr / np.min(np.diff(barkFreqList)))

x = w
hopSize = sr * 0.0025
nX = len(w)
nHop = getNFrame(nX, hopSize)

fftSize = int(np.ceil(sr / 40.0 * 4))#roundUpToPowerOf2(sr * 0.025)

print("Wait...")
snrList = np.zeros((nHop, nBark), dtype=np.float32)
snrSBList = np.zeros((nHop, nBark), dtype=np.float32)
ifList = np.zeros((nHop, nBark), dtype=np.float32)
for iFreq, freq in enumerate(barkFreqList):
  print(iFreq)
  rf = freq / sr
  ww = yang.createYangSNRParameter(rf)
  snrList[:, iFreq] = yang.calcYangSNR(w, rf, ww)[(np.arange(0, nHop, dtype=np.float64) * hopSize).round().astype(np.int32)]

  h, hd = yang.createYangIFParameter(rf, rf)
  for iHop in range(nHop):
    iCenter = iHop * hopSize

    frame = getFrame(w, iCenter, 1 + (ww.size - 1) * 3)
    snrSBList[iHop, iFreq] = yang.calcYangSNRSingleFrame(frame, rf, ww)

    frame = getFrame(w, iCenter, h.size)
    ifList[iHop, iFreq] = yang.calcYangIF(frame, h, hd)

pl.figure()
pl.title("SNR")
pl.imshow(snrList.T, cmap='plasma', interpolation='nearest', aspect='auto', origin='lower')
pl.figure()
pl.title("SNR SB")
pl.imshow(snrSBList.T, cmap='plasma', interpolation='nearest', aspect='auto', origin='lower')
pl.figure()
pl.title("IF")
pl.imshow(np.abs(ifList).T, cmap='plasma', interpolation='nearest', aspect='auto', origin='lower', vmin=0.0, vmax=0.05)

pl.show()
