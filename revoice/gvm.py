import numpy as np
import scipy.optimize as so
from .common import *
from . import lfmodel, hnm
import pylab as pl

def generateGlottalSource(nSample, tList, T0List, tpList, teList, taList, pulseFilterList, energyList, sr):
  assert tList.shape == T0List.shape == tpList.shape == teList.shape == taList.shape == (pulseFilterList.shape[0],) == energyList.shape
  assert pulseFilterList.shape[1] % 2 == 1

  (nPulse, nBin) = pulseFilterList.shape
  fftSize = (nBin - 1) * 2
  f = (np.arange(0, nBin, dtype=np.float32) / fftSize * sr).astype(np.float32)

  out = np.zeros(nSample, dtype=np.float32)
  EeList = np.zeros(nPulse, dtype=np.float32)
  for (iPulse, t) in enumerate(tList):
    T0 = T0List[iPulse]
    if T0 * sr > (fftSize // 2 - 1):
      raise ValueError("Frequency %f is too low, minimum acceptable is %f" % (1 / T0, sr / (fftSize // 2 - 1)))
    energy = energyList[iPulse]
    tp, te, ta = tpList[iPulse], teList[iPulse], taList[iPulse]
    pulseFilter = pulseFilterList[iPulse]
    fac = np.sqrt(np.mean(pulseFilter[1:] ** 2))
    if fac == 0.0:
      continue
    pulseFilter = pulseFilter / fac
    
    # create filtered glottal pulse
    iCenter = int((t + 0.5 * T0) * sr)
    tErr = (t + 0.5 * T0) - iCenter / sr

    fdPulse = np.zeros(nBin, dtype=np.complex64)
    fdPulse[1:] = lfmodel.calcSpectrum(f[1:], T0, 1.0, tp, te, ta)
    fdPulse += fdPulse * np.exp(-2j * T0 * np.pi * f)
    fdPulse *= np.exp(-2j * (tErr + fftSize / sr * 0.5 - T0) * np.pi * f)
    fdPulse *= pulseFilter
    tdPulse = np.fft.irfft(fdPulse)
    #pl.plot(tdPulse)
    #pl.show()

    # apply hanning window
    windowWidth = 2 * T0 * sr
    window = accurateHann(2 * T0 * sr, 0.5 * fftSize / windowWidth - 0.5, fftSize)
    tdPulse *= window
    tdPulse *= sr
    
    synthedEnergy = np.sum(tdPulse ** 2) / windowWidth * 4
    
    Ee = np.sqrt(energy / synthedEnergy)
    #print(np.sum(pulseFilter ** 2))
    #print(Ee)

    ob, oe, ib, ie = getFrameRange(nSample, iCenter, fftSize)
    out[ib:ie] += tdPulse[ob:oe] * Ee
    EeList[iPulse] = Ee

  return out, EeList

def generateNoisePart(nSample, tList, hopIdxList, T0List, EeList, tpList, teList, taList, vuvHopList, noiseHopEnvList, noiseHopEnergyList, hopSize, sr):
  assert tList.shape == T0List.shape == EeList.shape == tpList.shape == teList.shape == taList.shape

  (nHop,) = noiseHopEnergyList.shape
  openness = np.zeros(nSample, dtype=np.float32)

  for (iPulse, t) in enumerate(tList):
    iHop = hopIdxList[iPulse]
    noiseEnergy = noiseHopEnergyList[iHop]

    T0 = T0List[iPulse]
    Ee = EeList[iPulse]
    tp, te, ta = tpList[iPulse], teList[iPulse], taList[iPulse]
    periodLen = int(T0 * sr)

    st = np.linspace(0, periodLen / sr, periodLen, dtype=np.float32)
    tdOpenness = lfmodel.calcGlottalOpenness(st, T0, 1.0, tp, te, ta)
    tdOpenness += np.min(tdOpenness)
    tdOpenness /= np.max(tdOpenness)
    tdOpenness *= min(Ee ** 0.125, 1.0)

    window = np.hanning(periodLen * 2)
    windowNormFac = 1 / np.mean(window)
    td = np.tile(tdOpenness, 2)

    iCenter = int(t * sr) + periodLen
    ob, oe, ib, ie = getFrameRange(nSample, iCenter, periodLen * 2)
    openness[ib:ie] += td[ob:oe] * window[ob:oe]
  openness = 1.0 - openness

  noiseTemplate = np.random.uniform(-1.0, 1.0, nSample).astype(np.float32)
  noiseTemplate = hnm.filterNoise(noiseTemplate, noiseHopEnvList, hopSize)
  noiseTemplate *= openness
  '''
  pl.plot(openness)
  pl.plot(noiseTemplate)
  pl.show()
  '''
  del openness
  
  noise = np.zeros(nSample, dtype=np.float32)
  synthSize = int(2 * hopSize)
  if synthSize % 2 == 1:
    synthSize += 1
  window = np.hanning(synthSize).astype(np.float32)
  windowNormFac = 1 / np.mean(window)
  for iHop in range(nHop):
    noiseEnergy = noiseHopEnergyList[iHop]
    if noiseEnergy <= 0.0:
      continue
    iCenter = int(round(iHop * hopSize))
    noiseTemplateFrame = getFrame(noiseTemplate, iCenter, synthSize)
    # integrate template energy
    if noiseHopEnergyList is not None:
        noiseTemplateEnergy = np.mean((noiseTemplateFrame * window * windowNormFac) ** 2)
        # set energy
        if noiseTemplateEnergy > 0.0:
            windowedNoiseFrame = noiseTemplateFrame * np.sqrt(noiseEnergy / noiseTemplateEnergy) * window
        else:
            windowedNoiseFrame = np.zeros(synthSize, dtype=np.float32)
    else:
        windowedNoiseFrame = noiseTemplateFrame * window
    # ola
    ob, oe, ib, ie = getFrameRange(nSample, iCenter, synthSize)
    noise[ib:ie] += windowedNoiseFrame[ob:oe]
  return noise

@nb.njit()
def convertF0ListToTList(f0List, hopSize, sr):
  (nHop,) = f0List.shape
  tList = []
  hopIdxList = []

  t = 0.0
  while True:
    iHop = int(round(t * sr / hopSize))
    if iHop >= nHop:
      break
    f0 = f0List[iHop]
    if f0 <= 0.0:
      t += hopSize / sr
      continue
    tList.append(t)
    hopIdxList.append(iHop)
    t += 1 / f0
  return np.array(tList, dtype=np.float64), np.array(hopIdxList, dtype=np.int32)

def addDistortionEffect(distortionStrengthList, tList, T0List, tpList, teList, taList, pulseFilterList, energyList, sr):
  (nPulse,) = tList.shape
  tList = tList.copy()
  tpList, teList, taList = tpList.copy(), teList.copy(), taList.copy()
  energyList = energyList.copy()
  for iPulse in range(nPulse):
    strength = distortionStrengthList[iPulse]
    t = tList[iPulse]
    T0 = T0List[iPulse]
    energy = energyList[iPulse]
    tp, te, ta = tpList[iPulse], teList[iPulse], taList[iPulse]
    Fa, Rk, Rg = lfmodel.calcGFModelParameterFromLFModel(T0, tp, te, ta)

    osc = np.sin(2 * np.pi * t / T0 / 1.75)
    t += T0 * 0.1 * np.random.uniform(0.0, 1.0) * strength
    Fa *= 1.0 - osc * 0.8 * np.random.uniform(0.1, 1.0) * strength
    Rk *= 1.0 + osc * 0.6 * np.random.uniform(0.1, 1.0) * strength
    energy *= 1.0 - osc * 0.8 * np.random.uniform(0.1, 1.0) * strength

    tp, te, ta = lfmodel.calcLFModelParameterFromGFModel(T0, Fa, Rk, Rg)
    energyList[iPulse] = energy
    tpList[iPulse], teList[iPulse], taList[iPulse] = tp, te, ta
    tList[iPulse] = t
  pulseFilterList = pulseFilterList * np.exp(1j * distortionStrengthList.reshape(nPulse, 1) * np.random.uniform(-np.pi, np.pi, nPulse * pulseFilterList.shape[1]).reshape(*pulseFilterList.shape))
  pulseFilterList = pulseFilterList.astype(np.complex64)
  return tList, tpList, teList, taList, pulseFilterList, energyList
