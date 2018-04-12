from .common import *

@nb.jit(nb.float64(nb.float64[:], nb.float64, nb.float64, nb.float64), nopython = True, cache = True)
def analyze(x, freq, resolution, sr):
    nuttallCoeff = np.array((0.338946, 0.481973, 0.161054, 0.018027), dtype = np.float64)
    
    nResponse = int(np.ceil(4 * sr / freq))
    irr = np.zeros(nResponse)
    iri = np.zeros(nResponse)
    dirr = np.zeros(nResponse)
    diri = np.zeros(nResponse)

    k = np.arange(nuttallCoeff.shape[0])
    omega = 2 * np.pi * freq / sr
    omegaw = 2 * np.pi / nResponse
    for i in range(nResponse):
        sini = np.sin(omega * (i - nResponse / 2))
        cosi = np.cos(omega * (i - nResponse / 2))
        w = np.sum(nuttallCoeff * np.cos(k * omegaw * (i - nResponse / 2)))
        wd = np.sum(-omegaw * k * nuttallCoeff * np.sin(k * omegaw * (i - nResponse / 2)))
        irr[i] = w * cosi
        iri[i] = w * sini
        diri[i] = omega * w * cosi + wd * sini
        dirr[i] = wd * cosi - omega * w * sini

    yr = np.sum(irr * x)
    yi = np.sum(iri * x)
    ydr = np.sum(dirr * x)
    ydi = np.sum(diri * x)

    # Flanagan's equation
    deno = (yr * yr + yi * yi)
    if(deno != 0):
        return (yr * ydi - yi * ydr) / (yr * yr + yi * yi) / 2 / np.pi * sr
    else:
        return freq

def calcInputSize(freq, sr):
    return int(np.ceil(4 * sr / freq))

class Analyzer:
    def __init__(self, sr, freq, resolution, **kwargs):
        self.samprate = float(sr)
        self.freq = float(freq)
        self.resolution = float(resolution)

        assert freq > 0.0 and freq < self.samprate / 2
        assert resolution > 0.0 and resolution < self.samprate / 2

        self.nResponse = int(np.ceil(4 * self.samprate / self.freq))
        self.irr = np.zeros(self.nResponse)
        self.iri = np.zeros(self.nResponse)
        self.dirr = np.zeros(self.nResponse)
        self.diri = np.zeros(self.nResponse)

        # calculate impulse response(ir) and deterivate of impulse response(dir)
        nuttallCoeff = np.asarray((0.338946, 0.481973, 0.161054, 0.018027), dtype = np.float64)

        k = np.arange(nuttallCoeff.shape[0])
        omega = 2 * np.pi * freq / self.samprate
        omegaw = 2 * np.pi / self.nResponse
        for i in range(self.nResponse):
            sini = np.sin(omega * (i - self.nResponse / 2))
            cosi = np.cos(omega * (i - self.nResponse / 2))
            w = np.sum(nuttallCoeff * np.cos(k * omegaw * (i - self.nResponse / 2)))
            wd = np.sum(-omegaw * k * nuttallCoeff * np.sin(k * omegaw * (i - self.nResponse / 2)))
            self.irr[i] = w * cosi
            self.iri[i] = w * sini
            self.diri[i] = omega * w * cosi + wd * sini
            self.dirr[i] = wd * cosi - omega * w * sini
    
    def __call__(self, x):
        assert x.shape[0] == self.nResponse
        
        nX = x.shape[0]
        yr = np.sum(self.irr * x)
        yi = np.sum(self.iri * x)
        ydr = np.sum(self.dirr * x)
        ydi = np.sum(self.diri * x)

        # Flanagan's equation
        deno = (yr * yr + yi * yi)
        if(deno != 0):
            return (yr * ydi - yi * ydr) / (yr * yr + yi * yi) / 2 / np.pi * self.samprate
        else:
            return self.freq