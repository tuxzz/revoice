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

nHop = f0List.shape[0]

print("F0 Refinement...")
f0RefineProcessor = refinef0_stft.Processor(sr)
f0List = f0RefineProcessor(w, f0List)

print("HNM Analyzing...")
hnmAnalyzer = hnm.Analyzer(sr, harmonicAnalysisMethod = "qfft")
hFreqList, hAmpList, hPhaseList, sinusoidEnergyList, noiseEnvList, noiseEnergyList = hnmAnalyzer(w, f0List)

print("Rd Analyzing...")
nHop = f0List.shape[0]
rdList = np.zeros(nHop)
rdAnalyzer = rd_krh.Analyzer()
for iHop, f0 in enumerate(f0List):
    if(f0 <= 0):
        continue
    need = hFreqList[iHop] > 0
    rdList[iHop] = rdAnalyzer(hFreqList[iHop][need], hAmpList[iHop][need])

print("Voice tract Analyzing...")
hVtAmpList = np.zeros(hAmpList.shape)
for iHop, f0 in enumerate(f0List):
    if(f0 <= 0):
        continue
    need = hFreqList[iHop] > 0
    hFreq = hFreqList[iHop][need]
    hAmp = hAmpList[iHop][need]
    hPhase = hPhaseList[iHop][need]

    hVsAmp = np.abs(lfmodel.calcSpectrum(np.asarray(hFreq), 1 / f0, 1.0, *lfmodel.calcParameterFromRd(rdList[iHop])))
    hVsAmp /= hVsAmp[0]
    hVtAmpList[iHop][need] = hAmp / hVsAmp

print("Formant Analyzing...")
formantAnalyzer = formant_tkf.Analyzer(sr)
nFormant = 5
fFreqList = np.zeros((nHop, nFormant))
fBwList = np.zeros((nHop, nFormant))
fAmpList = np.zeros((nHop, nFormant))
for iHop, f0 in enumerate(f0List):
    if(f0 <= 0):
        continue
    need = hFreqList[iHop] > 0
    fFreqList[iHop], fBwList[iHop], fAmpList[iHop] = formantAnalyzer(hFreqList[iHop][need], hVtAmpList[iHop][need])

pl.plot((np.arange(nHop) + 0.5) * pyinAnalyzer.hopSize / sr, fFreqList)

pl.figure()
t = 0.46
iHop = int(t * sr / pyinAnalyzer.hopSize)
print(iHop)
need = hFreqList[iHop] > 0
pl.plot(hFreqList[iHop][need], np.log(hVtAmpList[iHop][need]))
pl.plot(hFreqList[iHop][need], np.log(calcKlattFilterBankResponseMagnitude(hFreqList[iHop][need], fFreqList[iHop], fBwList[iHop], fAmpList[iHop], sr)))

pl.show()
