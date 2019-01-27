import numpy as np
import scipy.io.wavfile as wavfile
import scipy.interpolate as ipl
import scipy.signal as sp
import scipy.special as spec
import scipy.optimize as so
import scipy.stats as ss
import scipy.linalg as sla
import numba as nb
import numbers

eps = 2.2204e-16
soundVelocity = 340.29 # m/s

windowDict = {
    # name: func(length), main-lobe-width, mean
    "hanning": (sp.hanning, 1.5, 0.5),
    "blackman": (sp.blackman, 1.73, 0.42),
    "blackmanharris": (sp.blackmanharris, 2.0044, (35875 - 3504 * np.pi) / 1e5),
}

def loadWav(path):
    """
    Load wave file into float64 ndarray

    Parameters
    ----------
    path: str
        Path of wave file
    
    Returns
    ----------
    tuple_like
        [0]: array_like
            Float64 wave data
            For mono data, shape = (nSample,)
            For multi channel data, shape = (nSample, nChannel)
        [1]: int
            Sample rate
    """
    samprate, w = wavfile.read(path)
    if(w.dtype == np.int8):
        w = w.astype(np.float64) / 127.0
    elif(w.dtype == np.short):
        w = w.astype(np.float64) / 32767.0
    elif(w.dtype == np.int32):
        w = w.astype(np.float64) / 2147483647.0
    elif(w.dtype == np.float32):
        w = w.astype(np.float64)
    elif(w.dtype == np.float64):
        pass
    else:
        raise ValueError("Unsupported sample format: %s" % (str(w.dtype)))
    return w, samprate

def saveWav(path, data, sr):
    """
    Save data to wave file.

    Parameters
    ----------
    path: str
        Path to save
    data: array_like
        Float wave data
        For mono data, shape = (nSample,)
        For multi channel data, shape = (nSample, nChannel)
    sr: int
        Sample rate of wave data
    """
    wavfile.write(path, int(sr), data)

def splitArray(seq, cond = lambda v:v <= 0.0 or np.isnan(v)):
    """
    Split array into multiple part using cond as condition.

    Parameters
    ----------
    seq: array_like, shape = (n,)
        Array to split
    cond: callable, optional
        Condition for splitting
    
    Returns
    ----------
    list
        List of parts
    """
    if(len(seq) == 0):
        return []
    o = []
    n = len(seq)
    last = 0
    i = 0
    while(i < n):
        if(cond(seq[i])):
            if(last != i):
                o.append(seq[last:i])
            last = i
            while(i < n and cond(seq[i])): i += 1
            o.append(seq[last:i])
            last = i
        i += 1
    if(last != n):
        o.append(seq[last:])
    return o

@nb.jit(nb.float64[:](nb.float64[:]), nopython = True, cache = True)
def removeDCSimple(x):
    """
    Simple DC remover based on subtract mean

    Parameters
    ----------
    x: array_like, shape = (n,)
        Input signal
    
    Returns
    ----------
    array_like, shape = (n,)
        Processed signal
    """
    return x - np.mean(x)

@nb.jit(nb.types.Tuple((nb.int64, nb.int64, nb.int64, nb.int64))(nb.int64, nb.int64, nb.int64), nopython = True, cache = True)
def getFrameRange(inputLen, center, size):
    """
    Calculate a range for getFrame

    Parameters
    ----------
    inputLen: int
        Length of input
    center: int
        Center position of frame
    size: int
        Length of frame
    
    Returns
    ----------
    tuple_like
        [0]: int
            Start position of output
        [1]: int
            End position of output
        [2]: int
            Start position of input
        [3]: int
            End position of input
    """
    leftSize = size // 2
    rightSize = size - leftSize # for odd size

    inputBegin = min(inputLen, max(center - leftSize, 0))
    inputEnd = max(0, min(center + rightSize, inputLen))

    outBegin = max(leftSize - center, 0)
    outEnd = outBegin + (inputEnd - inputBegin)

    return outBegin, outEnd, inputBegin, inputEnd

@nb.jit(nb.float64[:](nb.float64[:], nb.int64, nb.int64), nopython = True, cache = True)
def getFrame(input, center, size):
    """
    Get a frame from input

    Parameters
    ----------
    input: array_like
        Input array
    center: int
        Center position of frame
    size: int
        Length of frame
    
    Returns
    ----------
    array_like
        The frame
    """
    out = np.zeros(size, input.dtype)

    outBegin, outEnd, inputBegin, inputEnd = getFrameRange(len(input), center, size)

    out[outBegin:outEnd] = input[inputBegin:inputEnd]
    return out

@nb.jit(nb.int64(nb.int64, nb.float64), nopython = True, cache = True)
def getNFrame(inputSize, hopSize):
    """
    Calculate how many frame you can get

    Parameters
    ----------
    inputSize: int
        Length of input
    hopSize: real number
        Interval size between hops
    
    Returns
    ----------
    int
        nFrame
    """
    return int(np.ceil(inputSize / hopSize))

def getWindow(window):
    """
    Get a window object

    Parameters
    ----------
    window: str or tuple, len == 3
        For str:
            Window name
        For tuple, len == 3:
            (windowFunc, main-lobe-width, mean)
    
    Returns
    ----------
    int
        nFrame
    
    Raises
    ----------
    TypeError
        When specific window name not exists in windowDict
    """
    if(type(window) is str):
        return windowDict[window]
    elif(type(window) is tuple):
        assert len(window) == 3
        return window
    else:
        raise TypeError("Invalid window.")

def applyMovingAverageFilter(x, order):
    """
    Apply moving average filter on input signal

    Parameters
    ----------
    x: array_like, shape = (n,)
        Input signal
    order: int
        Length of moving average kernel
        Must be odd number
    
    Returns
    ----------
    array_like, shape = (n,)
        Processed signal
    """
    assert order % 2 == 1
    return sp.fftconvolve(x, np.full(order, 1.0 / order))[order // 2:order // 2 + len(x)]

def roundUpToPowerOf2(v):
    """
    Round a value up to the nearest power of 2

    Parameters
    ----------
    v: float
        Input value
    
    Returns
    ----------
    int
        Processed value
    """
    return int(2 ** np.ceil(np.log2(v)))

@nb.jit(nopython = True, cache = True)
def parabolicInterpolate(input, i, overAdjust = False):
    """
    Get interpolated peak using parabolic interpolation

    Parameters
    ----------
    input: array_like, shape = (n,)
        Input data
    i: int
        Position to interpolate
    overAdjust: bool, optional
        Allow distance of interpolated index to input index greater than 1.0
    
    Returns
    ----------
    tuple_like
        [0]: float
            Interpolated index
        [1]: float
            Interpolated value
    """
    lin = len(input)

    if(i > 0 and i < lin - 1):
        s0 = float(input[i - 1])
        s1 = float(input[i])
        s2 = float(input[i + 1])
        a = (s0 + s2) / 2.0 - s1
        if(a == 0):
            return (i, input[i])
        b = s2 - s1 - a
        adjustment = -(b / a * 0.5)
        if(not overAdjust and abs(adjustment) > 1.0):
            adjustment = 0.0
        x = i + adjustment
        y = a * adjustment * adjustment + b * adjustment + s1
        return (x, y)
    else:
        x = i
        return (x, y)

def fixComplexIntoUnitCircle(x):
    """
    If abs(x) is greater than 1.0
        Generate a new complex that abs(new_x) is equal to 1/abs(x), and keep the angle
    Otherwise copy it
    
    Parameters
    ----------
    x: complex or array_like, shape = (n,)
        Input data
    
    Returns
    ----------
    complex or array_like, shape = (n,)
        Fixed data
    """
    if(isinstance(x, complex)):
        return (1 + 0j) / np.conj(x) if np.abs(x) > 1.0 else x
    else:
        need = np.abs(x) > 1.0
        x[need] = (1 + 0j) / np.conj(x[need])
        return x

def formantFreq(n, L, c = soundVelocity):
    """
    Estimate format frequency for given format number based on vocal tract length

    [1] Watanabe, Akira. "Formant estimation method using inverse-filter control." IEEE Transactions on Speech and Audio Processing 9.4 (2001): 317-326.
    
    Parameters
    ----------
    n: int
        Number of format
    L: float
        Vocal tract length in meters
        For a adult male it may be 0.168
    c: float, optional
        Speed of sound in meters

    Returns
    ----------
    float
        Frequency in Hz
    """
    return (2 * n - 1) * c / 4 / L

def countFormant(freq, L, c = soundVelocity):
    """
    Estimate format number for given frequency based on vocal tract length

    [1] Watanabe, Akira. "Formant estimation method using inverse-filter control." IEEE Transactions on Speech and Audio Processing 9.4 (2001): 317-326.
    
    Parameters
    ----------
    freq: float
        Format frequency in Hz
    L: float
        Vocal tract length in meters
        For a adult male it may be 0.168
    c: float, optional
        Speed of sound in meters

    Returns
    ----------
    float:
        Number of formant
    """
    return (freq * 4 * L / c + 1) / 2

def getPreEmphasisFilterResponse(x, freq, sr):
    """
    Calculate response for pre-emphasis filter
    
    Parameters
    ----------
    x: array_like, shape = (n,)
        Frequency list
    freq: float
        Frequency for pre-emphasis filter
    sr: float
        Sample rate

    Returns
    ----------
    array_like, shape = (n,)
        Filter response for input frequency list
        Complex value
    """
    x = np.asarray(x)
    a = np.exp(-2.0 * np.pi * freq / sr)
    z = np.exp(2j * np.pi * x / sr)
    return 1 - a / z

def applyPreEmphasisFilter(x, freq, sr):
    """
    Apply a 1-order pre-emphasis filter to input signal
    
    Parameters
    ----------
    x: array_like, shape = (n,)
        Input signal
    freq: float
        Frequency for pre-emphasis filter
    sr: float
        Sample rate of input signal

    Returns
    ----------
    array_like, shape = (n,)
        Processed signal
    """
    o = np.zeros(len(x))
    fac = np.exp(-2.0 * np.pi * freq / sr)
    o[0] = x[0]
    o[1:] = x[1:] - x[:-1] * fac
    return o

def applyDeEmphasisFiler(x, freq, sr):
    """
    Apply an inverse filter of pre-emphasis filter to input signal
    
    Parameters
    ----------
    x: array_like, shape = (n,)
        Input signal
    freq: float
        Frequency for pre-emphasis filter
    sr: float
        Sample rate of input signal

    Returns
    ----------
    array_like, shape = (n,)
        Processed signal
    """
    o = x.copy()
    fac = np.exp(-2.0 * np.pi * freq / sr)

    for i in range(1, len(x)):
        o[i] += o[i - 1] * fac
    return o

def lerp(a, b, ratio):
    """
    Linear interpolate between a and b
    
    Parameters
    ----------
    a: number or array_like, shape = s
    b: number or array_like, shape = s
    ratio: float
        Linear interpolation ratio

    Returns
    ----------
    number or array_like, shape = s
        Interpolated data
    """
    return a + (b - a) * ratio

def freqToMel(x, a = 2595.0, b = 700.0):
    """
    Convert linear frequency to mel frequency

    m = a * log10(1 + f / b)
    where m is mel frequency, f is liner frequency
    
    Parameters
    ----------
    x: float or array_like, shape = (n,)
        Linear frequency
    a: float, optional
    b: float, optional

    Returns
    ----------
    float or array_like
        Mel frequency
    """
    return a * np.log10(1.0 + x / b)

def melToFreq(x, a = 2595.0, b = 700.0):
    """
    Convert mel frequency to linear frequency
    
    f = b * (10^(f / a) - 1.0)
    where m is mel frequency, f is liner frequency
    
    Parameters
    ----------
    x: float or array_like, shape = (n,)
        Mel frequency
    a: float, optional
    b: float, optional

    Returns
    ----------
    float or array_like
        Linear frequency
    """
    return (np.power(10, x / a) - 1.0) * b

def freqToPitch(freq):
    """
    Convert linear frequency to pitch

    p = log2(f / 440) * 12 + 69
    where p is pitch, f is liner frequency
    
    Parameters
    ----------
    x: float or array_like, shape = (n,)
        Linear frequency

    Returns
    ----------
    float or array_like
        Pitch
    """
    return np.log2(freq / 440.0) * 12.0 + 69.0

def pitchToFreq(semi):
    """
    Convert pitch to linear frequency

    f = 2^((p - 69) / 12) * 440
    where p is pitch, f is liner frequency
    
    Parameters
    ----------
    x: float or array_like, shape = (n,)
        Pitch

    Returns
    ----------
    float or array_like
        Linear frequency
    """
    return np.power(2, (semi - 69.0) / 12.0) * 440.0

def calcSRER(x, y):
    """
    Calculate SRER between two signals

    srer = log10(std(x) / std(x - y)) * 20.0
    
    Parameters
    ----------
    x: array_like, shape = (n,)
    y: array_like, shape = (n,)

    Returns
    ----------
    float:
        SRER value
    """
    return np.log10(np.std(x) / np.std(x - y)) * 20.0

@nb.jit(nopython = True, cache = True)
def sigmoid(x):
    """
    Sigmoid function

    y = 1 / (1 + e^(-x))
    
    Parameters
    ----------
    x: number or array_like

    Returns
    ----------
    number or array_like
    """
    return 1 / (1 + np.exp(-x))

@nb.jit(nopython = True, cache = True)
def lambertW(x):
    """
    Lambert W function
    Inverse function of f(x) = xe^x
    
    Parameters
    ----------
    x: number or array_like

    Returns
    ----------
    number or array_like
    """
    A = 2.344
    B = 0.8842
    C = 0.9294
    D = 0.5106
    E = -1.213
    y = (2.0 * np.e * x + 2.0) ** 0.5
    w = (2.0 * np.log(1.0 + B * y) - np.log(1.0 + C * np.log(1.0 + D * y)) + E) / (1.0 + 1.0 / (2.0 * np.log(1.0 + B * y) + 2.0 * A))
    for _ in range(24):
        u = np.exp(w)
        v = w * u - x
        w -= v / ((1.0 + w) * u - ((w + 2.0) * v) / (2.0 * w + 2.0))
    return w

def wrap(phase):
    """
    Inverse function of np.unwrap
    
    Parameters
    ----------
    phase: number or array_like

    Returns
    ----------
    number or array_like
    """
    out = phase - np.round(phase / (2.0 * np.pi)) * 2.0 * np.pi
    if(isinstance(phase, numbers.Number)):
        if(out > np.pi):
            out -= 2 * np.pi
        elif(out < np.pi):
            out += 2 * np.pi
    else:
        phase = np.asarray(phase)
        out[out > np.pi] -= 2 * np.pi
        out[out < -np.pi] += 2 * np.pi

    return out

def findPeak(x, lowerIdx, upperIdx):
    """
    Find peaks in array x
    
    Parameters
    ----------
    x: array_like
        Input array
    lowerIdx: int
        Start position for searching
    upperIdx: int
        End position for searching

    Returns
    ----------
    list
        A list of peak index
    """
    nBin = x.shape[0]
    assert(lowerIdx >= 0 and lowerIdx < nBin)
    assert(upperIdx >= 0 and upperIdx < nBin)

    if(lowerIdx >= upperIdx):
        return lowerIdx
    rcmp = x[lowerIdx + 1:upperIdx - 1]
    peakIdxList = np.arange(lowerIdx + 1, upperIdx - 1)[np.logical_and(np.greater(rcmp, x[lowerIdx:upperIdx - 2]), np.greater(rcmp, x[lowerIdx + 2:upperIdx]))]
    if(len(peakIdxList) == 0):
        return lowerIdx + np.argmax(x[lowerIdx:upperIdx])
    else:
        return peakIdxList[np.argmax(x[peakIdxList])]

def calcLogSpectrumMinphase(x):
    """
    Calculate minimum phase for even-length fft size log spectrum magnitude

    Parameters
    ----------
    x: array_like
        Log spectrum magnitude, shape = (nFFT // 2 + 1,)

    Returns
    ----------
    array_like
        Mimimum phase of input signal, shape = (nFFT // 2 + 1,)
    """
    
    nFFT = (x.shape[0] - 1) * 2
    cepstrum = np.fft.irfft(x.astype(np.complex128))

    cepstrum[1:nFFT // 2] *= 2
    return np.fft.rfft(cepstrum[:nFFT // 2 + 1], n = nFFT).imag

def calcSinusoidMinphase(hFreq, hAmp):
    nHar = hFreq.shape[0]
    nFFT = max(64, roundUpToPowerOf2(nHar + 2))
    nBin = nFFT // 2 + 1
    
    x = np.concatenate(((0,), hFreq)) / hFreq[0] / (nHar + 1) * nFFT * 0.5
    y = np.concatenate(((hAmp[0],), hAmp))
    magn = np.log(ipl.interp1d(x, y, kind = "linear", bounds_error = False, fill_value = y[-1])(np.arange(nBin)))
    phase = calcLogSpectrumMinphase(magn)
    
    out = ipl.interp1d(np.arange(nBin), phase, kind = "linear", bounds_error = True)(x[1:-1])
    return np.concatenate((out, (out[-1],)))

@nb.jit(nb.float64(nb.float64[:], nb.float64[:]), nopython = True, cache = True)
def calcItakuraSaitoDistance(p, phat):
    """
    Calculate Itakura-Saito distance between p and phat

    Parameters
    ----------
    p: array_like
        Usually be original spectrum, shape = (n,)
    phat: array_like
        Usually be estimated spectrum, shape = (n,)

    Returns
    ----------
    float
        Itakura-Saito distance between p and phat
    """
    assert (phat > 0.0).all(), "all values in second parameter must be greater than 0"
    r =  p / phat
    return np.log(np.mean(r - np.log(r) - 1))

def shiftPhase(shiftRatio, origPhase, theta):
    return wrap(origPhase + shiftRatio * theta)

def propagatePhase(hFreqList, hPhaseList, hopSize, sr, inversed):
    f0List = hFreqList[:, 0].copy()
    f0List[f0List < 0.0] = 0.0
    dList = np.cumsum(f0List) * (hopSize / sr * 2 * np.pi)
    o = hPhaseList.copy()
    for iHop, d in enumerate(dList):
        if(f0List[iHop] <= 0):
            continue
        if(inversed):
            d = -d
        o[iHop] = shiftPhase(hFreqList[iHop] / hFreqList[iHop][0], o[iHop], d)
    return o

@nb.jit(nb.complex128[:](nb.float64[:], nb.float64[:], nb.float64), nopython = True, cache = True)
def calcSpectrumAtFreq(x, freqList, sr):
    """
    Calculate specturm of x at specific frequency

    Parameters
    ----------
    x: array_like
        Input signal, shape = (nX,), nX must be even
    freqList: array_like
        A list of frequency to measure
    sr: float
        Samprate

    Returns
    ----------
    array_like
        Complex spectrum
    """
    assert x.shape[0] % 2 == 0

    nX = x.shape[0]
    nFreq = freqList.shape[0]
    
    t = np.arange(-nX // 2, nX // 2) / sr
    o = np.zeros(nFreq, dtype = np.complex128)
    n2jpit = -2j * np.pi * t
    for i, freq in enumerate(freqList):
        o[i] = np.sum(x * np.exp(n2jpit * freq))

    return o

def rfftFreq(nFFT, sr):
    return np.arange(nFFT // 2 + 1) / (nFFT // 2) * sr * 0.5

def minimizeScalar(costFunction, gridSearchPointList, args = ()):
    nGridSearchPoint = len(gridSearchPointList)
    assert nGridSearchPoint >= 2 or nGridSearchPoint == 0

    iBest = 0
    bestLoss = np.inf

    if(nGridSearchPoint == 2):
        return so.minimize_scalar(costFunction, args = args, method = "Bounded", bounds = gridSearchPointList).x
    elif(nGridSearchPoint == 0):
        return so.minimize_scalar(costFunction, args = args, method = "Brent").x
    else:
        # firstly perform grid search to prevent fall into local minima
        for i, x in enumerate(gridSearchPointList):
            loss = costFunction(x, *args)
            if(loss < bestLoss):
                iBest = i
                bestLoss = loss
        # perform bounded brent search
        i = iBest
        if(i == 0):
            bounds = (gridSearchPointList[0], gridSearchPointList[1])
        elif(i == len(gridSearchPointList) - 1):
            bounds = (gridSearchPointList[-2], gridSearchPointList[-1])
        else:
            bounds = (gridSearchPointList[i - 1], gridSearchPointList[i + 1])
        return so.minimize_scalar(costFunction, args = args, method = "Bounded", bounds = bounds).x

@nb.jit(nb.complex128[:](nb.complex128[:], nb.float64, nb.float64), nopython = True, cache = True)
def calcKlattFilterTransfer(z, bw, amp):
    C = -np.exp(2.0 * np.pi * bw)
    B = -2 * np.exp(np.pi * bw)
    A = 1 - B - C
    ampFac = (1.0 + B - C) / A * amp
    return A / (1.0 - B / z - C / (z * z)) * ampFac

@nb.jit(nb.complex128[:](nb.float64[:], nb.float64, nb.float64, nb.float64, nb.float64), nopython = True, cache = True)
def calcKlattFilterResponse(freqList, F, bw, amp, sr):
    z = np.exp(2.0j * np.pi * (0.5 + (freqList - F) / sr))
    return calcKlattFilterTransfer(z, bw / sr, amp)

@nb.jit(nb.float64[:](nb.float64[:], nb.float64[:], nb.float64[:], nb.float64[:], nb.float64), nopython = True, cache = True)
def calcKlattFilterBankResponseMagnitude(freqList, FList, bwList, ampList, sr):
    nFilter = FList.shape[0]
    assert bwList.shape[0] == ampList.shape[0]
    assert bwList.shape[0] == nFilter

    out = np.zeros(freqList.shape)
    for iFilter in range(nFilter):
        F, bw, amp = FList[iFilter], bwList[iFilter], ampList[iFilter]
        out += np.abs(calcKlattFilterResponse(freqList, F, bw, amp, sr))
    return out

@nb.jit(nb.int64(nb.int64, nb.int64), nopython = True, cache = True)
def calcCombination(n, r):
    result = 1.0
    if(r > n // 2):
        r = n - r
    deno = 1.0
    for i in range(1, r + 1):
        result *= n - i + 1
    for i in range(2, r + 1):
        deno *= i
    result /= deno
    return result

def iterCombination(n, r):
    assert r < n
    
    indexTable = np.arange(r, dtype = np.int)
    yield indexTable.copy()

    while True:
        yielded = False
        for i, index in enumerate(reversed(indexTable)):
            i = r - i - 1
            if(index < n - (r - i)):
                indexTable[i] += 1
                for offset, j in enumerate(range(i, r)):
                    indexTable[j] = index + offset + 1
                yield indexTable.copy()
                yielded = True
                break
        if(not yielded):
            return

def applySmoothingFilter(x, order):
    (nX,) = x.shape
    out = x.copy()
    if(nX < order):
        return out
    
    halfOrder = order // 2
    out[:halfOrder] = np.mean(x[:order])
    out[-halfOrder:] = np.mean(x[-order:])
    for i in range(halfOrder, nX - halfOrder):
        lowerIdx = i - halfOrder
        upperIdx = lowerIdx + order
        mean = np.mean(x[lowerIdx:upperIdx])
        nPos = np.sum(x[lowerIdx:upperIdx] >= mean)
        nNeg = np.sum(x[lowerIdx:upperIdx] <= mean)
        dTotal = np.sum(np.clip(x[lowerIdx:upperIdx] - mean, 0, np.inf))
        out[i] = mean + (nPos - nNeg) * dTotal / (order * order)
    
    return out