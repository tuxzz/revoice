from .common import *

def calcFormantFromLPC(coeff, sr):
    (order,) = coeff.shape
    #nMaxFormant = int(np.ceil(order / 2))
    nyq = sr * 0.5

    polyCoeff = np.zeros(order + 1)
    polyCoeff[:order] = coeff[::-1]
    polyCoeff[-1] = 1.0
    roots = np.roots(polyCoeff)
    roots = roots[roots.imag >= 0.0] # remove conjugate roots
    roots = fixComplexIntoUnitCircle(roots)
    
    nFormant = roots.shape[0]
    F = np.zeros(nFormant, dtype=np.float32)
    bw = np.zeros(nFormant, dtype=np.float32)

    for iRoot in range(nFormant):
        freq = np.abs(np.arctan2(roots.imag[iRoot], roots.real[iRoot])) * nyq / np.pi
        bandwidth = -np.log(np.abs(roots[iRoot]) ** 2) * nyq / np.pi
        F[iRoot] = freq
        bw[iRoot] = bandwidth

    sortOrder = np.argsort(F)
    F = F[sortOrder]
    bw = bw[sortOrder]

    return F, bw

def calcMagnitudeFromLPC(coeff, xms, fftSize, sr, deEmphasisFreq = None, bandwidthReduction = None):
    nyq = sr / 2
    nData = coeff.shape[0] + 1
    scale = 1.0 / np.sqrt(2.0 * nyq * nyq / (fftSize / 2.0))

    # copy to buffer
    fftBuffer = np.zeros(fftSize, dtype=np.float32)
    fftBuffer[0] = 1.0
    fftBuffer[1:nData] = coeff

    # deemphasis
    if(deEmphasisFreq is not None):
        fac = np.exp(-2 * np.pi * deEmphasisFreq / nyq)
        nData += 1
        for i in reversed(range(1, nData)):
            fftBuffer[i] -= fac * fftBuffer[i - 1]

    # reduce bandwidth
    if(bandwidthReduction is not None):
        fac = np.exp(np.pi * bandwidthReduction / sr)
        fftBuffer[1:nData] *= np.power(fac, np.arange(2, nData + 1))

    # do fft
    if xms > 0.0:
        scale *= np.sqrt(xms)
    o = np.fft.rfft(fftBuffer)
    o.real[0] = scale / o.real[0]
    o.imag[0] = 0.0
    o[1:fftSize // 2] = np.conj(o[1:fftSize // 2] * scale / (np.abs(o[1:fftSize // 2]) ** 2))
    o.real[-1] = scale / o.real[-1]
    o.imag[-1] = 0.0

    return np.abs(o, dtype=np.float32)

@nb.jit(nb.types.Tuple((nb.float32[:], nb.float32))(nb.float32[:], nb.int64), fastmath=True, nopython=True, cache=True)
def burgSingleFrame(x, order):
    n = x.shape[0]
    m = order

    a = np.ones(m, dtype=np.float32)
    aa = np.ones(m, dtype=np.float32)
    b1 = np.ones(n, dtype=np.float32)
    b2 = np.ones(n, dtype=np.float32)
    # (3)
    xms = np.sum(x * x) / n
    assert xms > 0

    # (9)
    b1[0] = x[0]
    b2[n - 2] = x[n - 1]
    b1[1:n - 1] = b2[:n - 2] = x[1:n - 1]

    for i in range(m):
        # (7)
        numer = np.sum(b1[:n - i - 1] * b2[:n - i - 1])
        deno = np.sum((b1[:n - i - 1] ** 2) + (b2[:n - i - 1] ** 2))
        if(deno <= 0):
            raise ValueError("Bad denominator (Is order too large for x?).")
        
        a[i] = 2.0 * numer / deno
        # (10)
        xms *= 1.0 - a[i] * a[i]
        # (5)
        a[:i] = aa[:i] - a[i] * aa[:i][::-1]
        if(i < m - 1):
            # (8)
            # NOTE: i -> i + 1
            aa[:i + 1] = a[:i + 1]
            for j in range(n - i - 2):
                b1[j] -= aa[i] * b2[j]
                b2[j] = b2[j + 1] - aa[i] * b1[j + 1]

    return -a, np.sqrt(xms * n)

def autocorrelationSingleFrame(x, order):
    n = x.shape[0]
    m = order

    # do autocorrelate via FFT
    nFFT = roundUpToPowerOf2(2 * n - 1)
    nx = np.min((m + 1, n))
    r = np.fft.irfft(np.abs(np.fft.rfft(x, n = nFFT) ** 2))
    r = r[:nx] / n
    a, _, _ = levinson1d(r, m)
    gain = np.sqrt(np.sum(a * r * n))

    return a[1:], gain

def slowAutocorrelationSingleFrame(x, order):
    n = x.shape[0]
    m = order

    p = m + 1
    r = np.zeros(p)
    nx = np.min((p, n))
    x = np.correlate(x, x, "full")
    r[:nx] = x[n - 1:n + m]
    a = np.dot(sla.pinv2(sla.toeplitz(r[:-1])), -r[1:])
    gain = np.sqrt(r[0] + np.sum(a * r[1:]))
    return a, gain

@nb.jit(nb.types.Tuple((nb.complex64[:], nb.float32, nb.complex64[:]))(nb.complex64[:], nb.int64), fastmath=True, nopython=True, cache=True)
def levinson1d(r, order):
    (n,) = r.shape
    assert n > 0, "r cannot be an empty array"
    assert order < n, "Order must be less than size"
    assert r[0].imag == 0.0, "First item of r must be real"
    assert r[0] != 0, "First item of r cannot be zero"

    # Estimated coefficients
    a = np.empty(order + 1, dtype=np.complex64)
    # temporary array
    t = np.empty(order + 1, dtype=np.complex64)
    # Reflection coefficients
    k = np.empty(order, dtype=np.complex64)

    a[0] = 1.0
    e = r[0].real

    for i in range(1, order + 1):
        acc = r[i]
        for j in range(1, i):
            acc += a[j] * r[i-j]
        k[i-1] = -acc / e
        a[i] = k[i-1]
        for j in range(order):
            t[j] = a[j]
        for j in range(1, i):
            a[j] += k[i-1] * np.conj(t[i-j])
        e *= 1 - k[i-1] * np.conj(k[i-1])

    return a, e.real, k

class Analyzer:
    supportedLPCAnalyzer = {
        "burg": burgSingleFrame,
        "ac": autocorrelationSingleFrame,
        "slowac": slowAutocorrelationSingleFrame,
    }

    def __init__(self, sr, **kwargs):
        self.samprate = sr
        self.hopSize = kwargs.get("hopSize", self.samprate * 0.0025)
        self.windowFac = kwargs.get("windowFac", 1.0)
        self.energyThreshold = kwargs.get("windowFac", 1e-8)
        self.order = kwargs.get("order", 8)
        self.lpcAnalysisMethod = kwargs.get("lpcAnalysisMethod", "burg")
        self.removeDC = kwargs.get("removeDC", True)
    
    def __call__(self, x, f0List, enableUnvoiced = True):
        (nX,) = x.shape
        (nHop,) = f0List.shape
        lpcAnalysisSingleFrame = self.supportedLPCAnalyzer[self.lpcAnalysisMethod]

        assert getNFrame(nX, self.hopSize) == nHop

        coeffList = np.zeros((nHop, self.order), dtype=np.float32)
        xmsList = np.zeros(nHop, dtype=np.float32)
        for iHop, f0 in enumerate(f0List):
            if not enableUnvoiced and f0 <= 0.0:
                continue
            iCenter = int(round(iHop * self.hopSize))

            windowSize = int((self.samprate / f0 * 4.0 if(f0 > 0.0) else self.hopSize * 6) * self.windowFac)
            if windowSize % 2 == 0:
                windowSize += 1
            frame = getFrame(x, iCenter, windowSize)
            if self.removeDC:
                frame = removeDCSimple(frame)

            if np.mean(frame ** 2) < self.energyThreshold:
                continue
            coeffList[iHop], xmsList[iHop] = lpcAnalysisSingleFrame(frame, self.order)

        return coeffList, xmsList