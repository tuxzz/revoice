import numpy as np
import pylab as pl
from revoice import *
from revoice.common import *

w, sr = loadWav("voices/yuri_orig.wav")
energyList = energy.Analyzer(sr)(w)

print("F0 Estimation...")
pyinAnalyzer = pyin.Analyzer(sr)
obsProbList = pyinAnalyzer(w)
monopitchAnalyzer = monopitch.Analyzer(*monopitch.parameterFromPYin(pyinAnalyzer))
f0List = monopitchAnalyzer(obsProbList)
silentList = energyList < 1e-8
f0List[silentList] = -np.abs(f0List[silentList])
hopSize = pyinAnalyzer.hopSize
nHop = f0List.shape[0]

# fix vuv flag before we have better vuv detector
# fix for voices/yuri_orig.wav
f0List[int(0.19 * sr / pyinAnalyzer.hopSize):int(0.354 * sr / pyinAnalyzer.hopSize) + 1] = np.abs(f0List[int(0.19 * sr / pyinAnalyzer.hopSize):int(0.354 * sr / pyinAnalyzer.hopSize) + 1])
f0List[int(2.669 * sr / pyinAnalyzer.hopSize):int(2.689 * sr / pyinAnalyzer.hopSize) + 1] = np.abs(f0List[int(2.669 * sr / pyinAnalyzer.hopSize):int(2.689 * sr / pyinAnalyzer.hopSize) + 1])
f0List[int(2.814 * sr / pyinAnalyzer.hopSize):int(2.86 * sr / pyinAnalyzer.hopSize) + 1] = np.abs(f0List[int(2.814 * sr / pyinAnalyzer.hopSize):int(2.86 * sr / pyinAnalyzer.hopSize) + 1])

print("F0 Refinement...")
f0RefineProcessor = refinef0_stft.Processor(sr)
f0List = f0RefineProcessor(w, f0List)

print("HNM Analyzing...")
hnmAnalyzer = hnm.Analyzer(sr, harmonicAnalysisMethod = "get")
hFreqList, hAmpList, hPhaseList, sinusoidEnergyList, noiseEnvList, noiseEnergyList = hnmAnalyzer(w, f0List)
assert hFreqList.dtype == hAmpList.dtype == hPhaseList.dtype == sinusoidEnergyList.dtype == noiseEnvList.dtype == noiseEnergyList.dtype == np.float32

print("Rd Analyzing...")
nHop = f0List.shape[0]
rdList = np.zeros(nHop, dtype=np.float32)
rdAnalyzer = rd_krh.Analyzer()
for iHop, f0 in enumerate(f0List):
    if(f0 <= 0):
        continue
    need = hFreqList[iHop] > 0
    rdList[iHop] = rdAnalyzer(hFreqList[iHop][need], hAmpList[iHop][need])

print("Split glottal and voice tract...")
envAnalyzer = mfienvelope.Analyzer(sr)
hVtAmpList = np.zeros(hAmpList.shape, dtype=np.float32)
for (iHop, f0) in enumerate(f0List):
  if f0 <= 0.0:
    continue
  hAmp = hAmpList[iHop]
  need = hFreqList[iHop] > 0
  hGlottalAmp = np.abs(lfmodel.calcSpectrum(hFreqList[iHop][need], 1 / f0, 1.0, *lfmodel.calcParameterFromRd(rdList[iHop])))
  normFac = hAmp[0] / hGlottalAmp[0]
  hGlottalAmp *= normFac
  hVtAmpList[iHop][need] = hAmp[need] / hGlottalAmp

print("Voice tract sinusoid synthing...")
synProc = hnm.Synther(sr)
sinusoid = synProc(hFreqList, hVtAmpList, hPhaseList, sinusoidEnergyList, None, None, enableNoise = False)
assert sinusoid.dtype == np.float32

print("MFI Envelope for voice tract sinusoid...")
envAnalyzer = mfienvelope.Analyzer(sr)
vtEnvList = envAnalyzer(sinusoid, f0List)
assert vtEnvList.dtype == np.float32

print("GVM Convert...")
(tList, hopIdxList) = gvm.convertF0ListToTList(f0List, hopSize, sr)
nPulse = tList.shape[0]
nSample = getNSample(nHop, hopSize)
T0List = 1 / f0List[hopIdxList]
tpList, teList, taList = np.zeros(nPulse, dtype=np.float32), np.zeros(nPulse, dtype=np.float32), np.zeros(nPulse, dtype=np.float32)
gvmEnergyList = energyList[hopIdxList]
gvmVtEnvList = np.exp(vtEnvList[hopIdxList])
for (iPulse, iHop) in enumerate(hopIdxList):
  (tpList[iPulse], teList[iPulse], taList[iPulse]) = lfmodel.calcParameterFromRd(rdList[iHop])

print("GVM Synth...")
osr = 1
if osr > 1:
  l = np.zeros((nPulse, (gvmVtEnvList.shape[1] - 1) * osr + 1), dtype=np.float32)
  for i, x in enumerate(gvmVtEnvList):
    l[i, :gvmVtEnvList.shape[1]] = x
    del i, x
  gvmVtEnvList = l
gvmOut, EeList = gvm.generateGlottalSource(nSample * osr, tList, T0List, tpList, teList, taList, gvmVtEnvList, gvmEnergyList, sr * osr)
if osr > 1:
  gvmOut = sp.resample_poly(gvmOut, 1, osr).astype(np.float32)
saveWav("gvm_basic_glottal.wav", gvmOut, sr)
gvmNoiseOut = gvm.generateNoisePart(nSample, tList, hopIdxList, T0List, EeList, tpList, teList, taList, f0List > 0.0, noiseEnvList, noiseEnergyList, hopSize, sr)
saveWav("gvm_basic_noise.wav", gvmNoiseOut, sr)

saveWav("gvm_basic_combined.wav", gvmOut + gvmNoiseOut, sr)
assert gvmOut.dtype == gvmNoiseOut.dtype == np.float32

print("Make growl effect...")
for (iPulse, iHop) in enumerate(hopIdxList):
  (tpList[iPulse], teList[iPulse], taList[iPulse]) = lfmodel.calcParameterFromRd(rdList[iHop] * 0.5)
tList, tpList, teList, taList, gvmVtEnvList, gvmEnergyList = gvm.addDistortionEffect(np.ones(nPulse, dtype=np.float32), tList, T0List, tpList, teList, taList, gvmVtEnvList, gvmEnergyList, sr)

print("GVM synth for growl...")
gvmOut, EeList = gvm.generateGlottalSource(nSample * osr, tList, T0List, tpList, teList, taList, gvmVtEnvList, gvmEnergyList, sr * osr)
if osr > 1:
  gvmOut = sp.resample_poly(gvmOut, 1, osr).astype(np.float32)
saveWav("gvm_basic_growl_glottal.wav", gvmOut, sr)
gvmNoiseOut = gvm.generateNoisePart(nSample, tList, hopIdxList, T0List, EeList, tpList, teList, taList, f0List > 0.0, noiseEnvList, noiseEnergyList, hopSize, sr)
saveWav("gvm_basic_growl_noise.wav", gvmNoiseOut, sr)

saveWav("gvm_basic_growl_combined.wav", gvmOut + gvmNoiseOut, sr)
assert gvmOut.dtype == gvmNoiseOut.dtype == np.float32