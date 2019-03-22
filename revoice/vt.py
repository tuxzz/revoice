from .common import *
import pylab as pl

@nb.jit()
def areaToReflection(a: np.ndarray) -> np.ndarray:
  out = np.zeros(a.shape, dtype=np.float32)
  for i in range(1, a.shape[0]):
    if a[i] == 0.0:
      out[i] = 0.9999
    else:
      out[i] = (a[i - 1] - a[i]) / (a[i - 1] + a[i])
  return out

@nb.jit()
def reflectionToALPCoeff(r: np.ndarray) -> np.ndarray:
  n = r.shape[0] - 1
  a = np.zeros(n + 1, dtype=np.float32)
  b = np.zeros(n, dtype=np.float32)
  a[0] = -1
  a[1] = r[1]
  for i in range(1, n):
    for j in range(i):
      b[j] = a[j + 1] - r[i + 1] * a[i - j]
    a[1:] = b
    a[i + 1] = r[i + 1]
  return -a

# Equal Length Tube Model using ALP
@nb.jit()
def applyVoiceTractFilterSingleTractALP(x: np.ndarray, tractAreaList: np.ndarray, glottalReflection: float, lipReflection: float) -> np.ndarray:
  reflection = areaToReflection(tractAreaList)
  f = reflectionToALPCoeff(np.concatenate(([glottalReflection], reflection, [lipReflection])))
  return sp.lfilter([1.0], f, x).astype(np.float32)

@nb.jit()
def calcVoiceTractSectionCount(vtLen: float, sr: float, c: float = soundVelocity) -> int:
  return int(round((vtLen * sr) / c))

@nb.jit()
def calcVoiceTractLength(nSection: int, sr: float, c: float = soundVelocity) -> float:
  return nSection * c / sr

@nb.jit()
def calcVoiceTractRadius(posList: np.ndarray, radiusList: np.ndarray, sr: float, c: float = soundVelocity) -> np.ndarray:
  assert posList.ndim == 1, "input array must be 1d array"
  assert posList.shape == radiusList.shape, "shape of all input arrays must be equal"
  assert posList.shape[0] > 1, "input length must be greater than 1"
  assert (np.diff(posList) > 0).all(), "posList must be ascending order"
  
  if posList[0] > 0.0:
    posList = np.concatenate(([0.0], posList))
    radiusList = np.concatenate(([radiusList[0]], radiusList))
  
  vtLen = posList[-1]
  nSection = calcVoiceTractSectionCount(vtLen, sr, c)

  return ipl.interp1d(posList, radiusList, kind="linear")(np.linspace(0.0, posList[-1], nSection)).astype(np.float32)

# Equal Length Tube Model
@nb.jit()
def applyVoiceTractFilterSingleTract(x: np.ndarray, tractAreaList: np.ndarray, glottalReflection: float, lipReflection: float) -> np.ndarray:
  reflection = areaToReflection(tractAreaList)
  return applyVoiceTractFilterSingleTractByReflection(x, reflection, glottalReflection, lipReflection)

@nb.jit()
def applyVoiceTractFilterSingleTractByReflection(x: np.ndarray, reflection: np.ndarray, glottalReflection: float, lipReflection: float) -> np.ndarray:
  (nSection,) = reflection.shape

  junctionOutputR = np.zeros(nSection + 1)
  junctionOutputL = np.zeros(nSection + 1)
  R = np.zeros(nSection)
  L = np.zeros(nSection)

  out = np.zeros(x.shape, dtype=np.float32)
  for i in range(x.shape[0]):
    junctionOutputR[0] = L[0] * glottalReflection + x[i]
    junctionOutputL[nSection] = R[nSection-1] * lipReflection
    for j in range(1, nSection):
      w = reflection[j] * (R[j - 1] + L[j])
      junctionOutputR[j] = R[j - 1] - w
      junctionOutputL[j] = L[j] + w
    for j in range(nSection):
      R[j] = junctionOutputR[j] * 0.99999
      L[j] = junctionOutputL[j + 1] * 0.99999
    out[i] += R[nSection - 1]
  return out

# Equal Length Tube Model with Nasal
@nb.jit()
def applyVoiceTractFilterTwoTract(x: np.ndarray, mainTractAreaList: np.ndarray, noseTractAreaList: np.ndarray, velumArea: float, glottalReflection: float, lipReflection: float) -> np.ndarray:
  (nMainSection,) = mainTractAreaList.shape
  (nNoseSection ,) = noseTractAreaList.shape
  assert nMainSection > 1 and nNoseSection > 1, "section count muse be greater than 1"
  assert nMainSection > nNoseSection, "section count of nose tract must be less than main tract"

  mainReflection = areaToReflection(mainTractAreaList)
  noseReflection = areaToReflection(noseTractAreaList)

  mainJunctionOutputR = np.zeros(nMainSection)
  mainJunctionOutputL = np.zeros(nMainSection + 1)
  R = np.zeros(nMainSection)
  L = np.zeros(nMainSection)

  noseJunctionOutputR = np.zeros(nNoseSection)
  noseJunctionOutputL = np.zeros(nNoseSection + 1)
  noseR = np.zeros(nNoseSection)
  noseL = np.zeros(nNoseSection)
  iNoseStart = nMainSection - nNoseSection

  sumArea = mainTractAreaList[iNoseStart] + mainTractAreaList[iNoseStart + 1] + velumArea
  leftReflection = (2 * mainTractAreaList[iNoseStart] - sumArea) / sumArea
  rightReflection = (2 * mainTractAreaList[iNoseStart + 1] - sumArea) / sumArea
  noseJunctionReflection = (2 * velumArea - sumArea) / sumArea

  out = np.zeros(x.shape, dtype=np.float32)
  for i in range(x.shape[0]):
    # Main Tract
    mainJunctionOutputR[0] = L[0] * glottalReflection + x[i]
    mainJunctionOutputL[nMainSection] = R[nMainSection-1] * lipReflection
    for j in range(1, nMainSection):
      w = mainReflection[j] * (R[j - 1] + L[j])
      mainJunctionOutputR[j] = R[j - 1] - w
      mainJunctionOutputL[j] = L[j] + w
    # Nose Tract Junction
    mainJunctionOutputL[iNoseStart] = leftReflection * R[iNoseStart - 1] + (1 + leftReflection) * (noseL[0] + L[iNoseStart])
    mainJunctionOutputR[iNoseStart] = rightReflection * L[iNoseStart] + (1 + rightReflection) * (R[iNoseStart - 1] + noseL[0])
    noseJunctionOutputR[0] = noseJunctionReflection * noseL[0] + (1 + noseJunctionReflection) * (L[iNoseStart] + R[iNoseStart - 1])
    # Rest Step of Main Tract
    for j in range(nMainSection):
      R[j] = mainJunctionOutputR[j] * 0.99999
      L[j] = mainJunctionOutputL[j + 1] * 0.99999
    # Nose Tract
    noseJunctionOutputL[nNoseSection] = noseR[nNoseSection - 1] * lipReflection
    for j in range(1, nNoseSection):
      w = noseReflection[j] * (noseR[j - 1] + noseL[j])
      noseJunctionOutputR[j] = noseR[j - 1] - w
      noseJunctionOutputL[j] = noseL[j] + w
    for j in range(nNoseSection):
      noseR[j] = noseJunctionOutputR[j] * 0.99999
      noseL[j] = noseJunctionOutputL[j + 1] * 0.99999
    # Output
    out[i] += R[nMainSection - 1] + noseR[nNoseSection - 1]
  return out