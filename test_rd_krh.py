import numpy as np
import pylab as pl
from revoice import *
from revoice.common import *

w, sr = loadWav("voices/kurumi_01.wav")
energyList = energy.Analyzer(sr)(w)

print("F0 Estimation...")
pyinAnalyzer = pyin.Analyzer(sr)
obsProbList = pyinAnalyzer(w)
monopitchAnalyzer = monopitch.Analyzer(*monopitch.parameterFromPYin(pyinAnalyzer))
f0List = monopitchAnalyzer(obsProbList)
silentList = energyList < 1e-8
f0List[silentList] = -np.abs(f0List[silentList])

print("F0 Refinement...")
f0RefineProcessor = refinef0_stft.Processor(sr)
f0List = f0RefineProcessor(w, f0List)

print("HNM Analyzing...")
hnmAnalyzer = hnm.Analyzer(sr, harmonicAnalysisMethod = "qfft")
hFreqList, hAmpList, hPhaseList, sinusoidEnergyList, noiseEnvList, noiseEnergyList = hnmAnalyzer(w, f0List)

print("Rd Analyzing...")
nHop = f0List.shape[0]
rdList = np.zeros(nHop, dtype=np.float32)
rdAnalyzer = rd_krh.Analyzer()
for iHop, f0 in enumerate(f0List):
    if(f0 <= 0):
        continue
    need = hFreqList[iHop] > 0
    rdList[iHop] = rdAnalyzer(hFreqList[iHop][need], hAmpList[iHop][need])

pl.plot((np.arange(w.shape[0]) + 0.5) / sr, w)
pl.plot((np.arange(nHop) + 0.5) * pyinAnalyzer.hopSize / sr, rdList)

pl.figure()
t = 0.46
iHop = int(t * sr / pyinAnalyzer.hopSize)
need = hFreqList[iHop] > 0
pl.plot(hFreqList[iHop][need], np.log(hAmpList[iHop][need]))
lfAmp = np.abs(lfmodel.calcSpectrum(hFreqList[iHop][need], 1 / hFreqList[iHop][0], 1, *lfmodel.calcParameterFromRd(rdList[iHop])))
lfAmp *= np.sqrt((hAmpList[iHop][0] ** 2) / (lfAmp[0] ** 2))
pl.plot(hFreqList[iHop][need], np.log(lfAmp))

pl.show()
