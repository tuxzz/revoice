from .common import *

# [1] Watanabe, Akira. "Formant estimation method using inverse-filter control." IEEE Transactions on Speech and Audio Processing 9.4 (2001): 317-326.
# [1] (2.2)

def ifcBasicFilterTransfer(z, B, F):
    alpha = np.exp(-2 * np.pi * B)
    beta = -2 * np.exp(-np.pi * B) * np.cos(2 * np.pi * F)
    gamma = 1 / (1 + alpha + beta)
    return gamma * (1 + beta / z + alpha / (z * z))

def ifcBasicFilterResponse(freqList, B, F, sr):
    return ifcBasicFilterTransfer(np.exp(freqList / sr * 2j * np.pi), B / sr, F / sr)

def costFunction(x, hFreq, hEnergy, sr):
    fFreq, fBw, fAmp = x.reshape(3, x.shape[0] // 3)
    fEnergy = calcKlattFilterBankResponseMagnitude(hFreq, fFreq, fBw, fAmp, sr)
    fEnergy *= fEnergy
    return calcItakuraSaitoDistance(hEnergy, fEnergy)

class Analyzer:
    def __init__(self, sr, **kwargs):
        self.samprate = sr
        self.voiceTractLength = kwargs.get("voiceTractLength", 0.15)
    
    def __call__(self, hFreq, hAmp):
        need = np.logical_and(hFreq > 0.0, hFreq <= 6000.0)
        hFreq = hFreq[need]
        hAmp = hAmp[need]
        fFreq = formantFreq(np.arange(1, 5 + 1), self.voiceTractLength)
        fBw = np.full(fFreq.shape, 300.0)
        fAmp = ipl.interp1d(np.concatenate(((0,), hFreq)), np.concatenate(((hAmp[0],), hAmp)), kind = "linear", bounds_error = False, fill_value = hAmp[-1])(fFreq)
        
        hEnergy = hAmp * hAmp
        
        fBound = np.concatenate((np.full(5, 100), np.full(5, 6000))).reshape(5, 2)
        bwBound = np.concatenate((np.full(5, 100), np.full(5, 400))).reshape(5, 2)
        ampBound = np.concatenate((np.full(5, 1e-8), np.full(5, 5.0))).reshape(5, 2)
        bound = np.concatenate((fBound, bwBound, ampBound))
        x = so.minimize(costFunction, np.array((fFreq, fBw, fAmp)), args = (hFreq, hEnergy, self.samprate), method = "L-BFGS-B", bounds = bound).x
        fFreq, fBw, fAmp = x.reshape(3, x.shape[0] // 3)
        return fFreq, fBw, fAmp
