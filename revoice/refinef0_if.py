from .common import *
from . import instantfrequency

class Processor:
    def __init__(self, sr, **kwargs):
        self.samprate = float(sr)
        self.hopSize = kwargs.get("hopSize", self.samprate * 0.0025)
        self.nHar = kwargs.get("nHar", 3)
        self.maxAdjustmentRatio = kwargs.get("maxAdjustmentRatio", 0.1)
        self.removeDC = kwargs.get("removeDC", True)
    
    def __call__(self, x, f0List):
        nHop = getNFrame(x.shape[0], self.hopSize)
        assert (nHop,) == f0List.shape

        out = f0List.copy()
        for iHop, f0 in enumerate(f0List):
            if(f0 <= 0.0):
                continue
            iCenter = int(round(iHop * self.hopSize))

            fAvgList = []
            for iHar in range(1, self.nHar + 1):
                frame = getFrame(x, iCenter, instantfrequency.calcInputSize(f0 * iHar, self.samprate))
                if(self.removeDC):
                    frame = removeDCSimple(frame)
                f = instantfrequency.analyze(frame, f0 * iHar, f0, self.samprate) / iHar
                if(np.abs(f - f0) < f0 * self.maxAdjustmentRatio):
                    fAvgList.append(f)
            if(fAvgList):
                out[iHop] = np.mean(fAvgList)
        
        return out