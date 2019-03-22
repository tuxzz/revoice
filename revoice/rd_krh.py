from .common import *
from . import lfmodel

# Kanru Hua's magnitude-based Rd estimator
# [1] https://github.com/Sleepwalking/prometheus-spark/blob/master/writings/pseudo-glottal-inverse-filter-hua-2016.pdf

@nb.jit(nb.float32(nb.float32, nb.float32[:], nb.float32[:]), fastmath=True, nopython=True, cache=True)
def costFunction(rd, hFreq, hEnergy):
    lfSpec = lfmodel.calcSpectrum(hFreq, 1.0 / hFreq[0], 1, *lfmodel.calcParameterFromRd(rd))

    lfEnergy = np.abs(lfSpec)
    lfEnergy *= lfEnergy
    gain = hEnergy[0] / lfEnergy[0]
    lfEnergy *= gain

    errMagn = np.exp(calcItakuraSaitoDistance(hEnergy, lfEnergy))

    return errMagn

class Analyzer:
    def __init__(self, **kwargs):
        self.gridSearchPointList = kwargs.get("gridSearchPointList", np.linspace(0.02, 3, 32))
        self.maxHarmonicFreq = kwargs.get("maxHarmonicFreq", 8000.0)
    
    def __call__(self, hFreq, hAmp):
        need = np.logical_and(hFreq > 0, hFreq <= self.maxHarmonicFreq)
        hFreq = hFreq[need]
        hAmp = hAmp[need]
        hEnergy = hAmp * hAmp

        return minimizeScalar(costFunction, self.gridSearchPointList, (hFreq, hEnergy))
