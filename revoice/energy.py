from .common import *

class Analyzer:
    def __init__(self, sr, **kwargs):
        self.samprate = float(sr)
        self.hopSize = kwargs.get("hopSize", self.samprate * 0.0025)
        self.removeDC = kwargs.get("removeDC", True)
    
    def __call__(self, x):
        nX = x.shape[0]
        nHop = getNFrame(nX, self.hopSize)

        out = np.zeros(nHop)
        frameSize = int(2 * self.hopSize)
        if(frameSize % 2 == 1):
            frameSize += 1
        for iHop in range(nHop):
            iCenter = int(round(iHop * self.hopSize))
            frame = getFrame(x, iCenter, frameSize)
            if(self.removeDC):
                frame = removeDCSimple(frame)
            out[iHop] = np.mean(frame ** 2)
        return out