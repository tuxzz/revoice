import numpy as np
import pylab as pl
from revoice import *
from revoice.common import *
import scipy.fftpack as sfft

w, sr = loadWav("voices/yuri_orig.wav")
w = sp.resample_poly(w, int(np.round(w.shape[0] / sr * 12000.0)), w.shape[0])
sr = 12000.0

w = applyPreEmphasisFilter(w, 50.0, sr)

nMaxFormant = 4
order = nMaxFormant * 2 + 1
fftSize = 4096

print("PYin...")
pyinAnalyzer = pyin.Analyzer(sr)
obsProbList = pyinAnalyzer(w)
energyList = energy.Analyzer(sr)(w)
nHop = energyList.shape[0]

print("Monopitch...")
monopitchAnalyzer = monopitch.Analyzer(*monopitch.parameterFromPYin(pyinAnalyzer))
f0List = monopitchAnalyzer(obsProbList)
silentList = energyList < 1e-8
f0List[silentList] = -np.abs(f0List[silentList])

print("LPC Analyzing...")
lpcProc = lpc.Analyzer(sr, order = order)
coeffList, xmsList = lpcProc(w, f0List)

lpcSpectrum = np.zeros((nHop, 2049))
FList, bwList = np.zeros((nHop, int(np.ceil(order * 0.5)))), np.zeros((nHop, int(np.ceil(order * 0.5))))
for iHop, f0 in enumerate(f0List):
    lpcSpectrum[iHop] = lpc.calcMagnitudeFromLPC(coeffList[iHop], xmsList[iHop], fftSize, sr, deEmphasisFreq = 50.0)
    F, bw = lpc.calcFormantFromLPC(coeffList[iHop], sr)
    need = np.logical_and(F > 50.0, F < sr * 0.5)
    F, bw = F[need], bw[need]
    FList[iHop, :F.shape[0]], bwList[iHop, :F.shape[0]] = F, bw

FList[f0List <= 0] = np.nan
FList[FList <= 0] = np.nan

lpcSpectrum = np.log(np.clip(lpcSpectrum, 1e-6, np.inf))
tList = np.arange(f0List.shape[0]) * pyinAnalyzer.hopSize / sr
pl.imshow(lpcSpectrum.T, interpolation = 'bicubic', aspect = 'auto', origin = 'lower', extent = [tList[0], tList[-1], 0, sr / 2])
pl.plot(tList, FList, 'o')
pl.show()
