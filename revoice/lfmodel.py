import numpy as np
import scipy.optimize as so
from .common import *
import pylab as pl

# [1] Huber, Stefan, and Axel Roebel. "On the use of voice descriptors for glottal source shape parameter estimation." Computer Speech & Language 28.5 (2014): 1170-1194.
# [2] Fant, Gunnar. "The LF-model revisited. Transformations and frequency domain analysis." Speech Trans. Lab. Q. Rep., Royal Inst. of Tech. Stockholm 2.3 (1995): 40.
@nb.jit(nb.types.Tuple((nb.float64, nb.float64, nb.float64))(nb.float64), nopython = True, cache = True)
def calcParameterFromRd(Rd):
    """
    Calculate tp, te, ta from Rd

    Parameters
    ----------
    Rd: float
        LF model shape parameter

    Returns
    ----------
    tuple
        float, float float
            tp, te, ta
    """
    if(Rd < 0.21): Rap = 1e-6
    elif(Rd < 2.7): Rap = (4.8 * Rd - 1.0) * 0.01
    else: Rap = 0.323 / Rd
    if(Rd < 2.7):
        Rkp = (22.4 + 11.8 * Rd) * 0.01
        Rgp = 0.25 * Rkp / ((0.11 * Rd) / (0.5 + 1.2 * Rkp) - Rap)
    else:
        OQupp = 1.0 - 1.0 / (2.17 * Rd)
        Rgp = 9.3552e-3 + 5.96 / (7.96 - 2.0 * OQupp)
        Rkp = 2.0 * Rgp * OQupp - 1.0428

    tp = 1.0 / (2.0 * Rgp)
    te = tp * (Rkp + 1.0)
    ta = Rap
    return tp, te, ta

# [1] Fant, Gunnar, Johan Liljencrants, and Qi-guang Lin. "A four-parameter model of glottal flow." STL-QPSR 4.1985 (1985): 1-13.
# [2] Doval, Boris, Christophe d'Alessandro, and Nathalie Henrich. "The spectrum of glottal flow models." Acta acustica united with acustica 92.6 (2006): 1026-1046.
@nb.jit(nb.types.Tuple((nb.float64, nb.float64, nb.float64, nb.float64, nb.float64, nb.float64, nb.float64))(nb.float64, nb.float64, nb.float64, nb.float64, nb.float64), nopython = True, cache = True)
def calcInternalParameter(T0, Ee, tp, te, ta):
    """
    Calculate wg, sinWgTe, cosWgTe, e, A, a, E0 from T0, Ee, tp, te, ta

    Parameters
    ----------
    T0, Ee, tp, te, ta: float, float, float, float, float
        LF Model parameters, tp, te, ta are T0-normalized

    Returns
    ----------
    tuple
        float, float, float, float, float, float, float
            wg, sinWgTe, cosWgTe, e, A, a, E0
    """
    assert T0 > 0.0 and Ee > 0.0 and tp > 0.0 and te > 0.0 and ta > 0.0
    wg = np.pi / tp #[1](2)
    sinWgTe = np.sin(wg * te)
    cosWgTe = np.cos(wg * te)

    e = (-ta * lambertW((te - T0) * np.exp((te - T0) / ta) / ta) + te - T0) / (ta * te - ta * T0) # [1] (12)
    A = e * ta / (e * e * ta) + (te - T0) * (1.0 - e * ta) / (e * ta) # [3] p.18, integral{0, T0} Ug(t) dt

    a = wg * (A * sinWgTe * wg - cosWgTe + 1)
    for iIter in range(8):
        a = a - ((a * a + wg * wg) * sinWgTe * A + wg * (np.exp(-a * te) - cosWgTe) + a * sinWgTe) / (sinWgTe * (2 * A * a + 1) - wg * te * np.exp(-te * a))
    '''
    afunc = lambda x : (x * x + wg * wg) * sinWgTe * A + wg * (np.exp(-x * te) - cosWgTe) + x * sinWgTe
    dafunc = lambda x : sinWgTe * (2 * A * x + 1) - wg * te * np.exp(-te * x)
    aa = so.fsolve(afunc, 0.0, fprime = dafunc)[0]
    assert (a - aa) / aa < 1e-10
    '''

    assert a < 1e9

    E0 = -Ee / (np.exp(a * te) * sinWgTe) # (5)
    return wg, sinWgTe, cosWgTe, e, A, a, E0

# [1] Doval, Boris, Christophe d'Alessandro, and Nathalie Henrich. "The spectrum of glottal flow models." Acta acustica united with acustica 92.6 (2006): 1026-1046.
@nb.jit(nb.complex128[:](nb.float64[:], nb.float64, nb.float64, nb.float64, nb.float64, nb.float64), nopython = True, cache = True)
def calcSpectrum(f, T0, Ee, tp, te, ta):
    """
    Calculate spectrum form given lf model parameters

    Parameters
    ----------
    f: array_like
        List of frequency
    sr: float
        Samprate, used for pulse amplitude normalization
        If you don't need amplitude normalization, you can just set it to 1.0
    T0: float
        Fundamental period, 1 / f0
    Ee: float
        Pulse amplitude
    tp, te, ta: float, float, float
        LF model shape parameters, T0-normalized

    Returns
    ----------
    array_like
        Spectrum, shape = f.shape
    """
    assert (f > 0.0).all()
    tp *= T0
    te *= T0
    ta *= T0
    wg, sinWgTe, cosWgTe, e, A, a, E0 = calcInternalParameter(T0, Ee, tp, te, ta)

    r = a - 2.0j * np.pi * f
    P1 = E0 / (r * r + wg * wg)
    P2 = wg + np.exp(r * te) * (r * sinWgTe - wg * cosWgTe)
    P3 = Ee * np.exp((-2.0j * np.pi * te) * f) / ((2.0j * e * ta * np.pi * f) * (e + 2.0j * np.pi * f))
    P4 = e * (1.0 - e * ta) * (1.0 - np.exp((-2.0j * np.pi * (T0 - te)) * f)) - (2.0j * e * ta * np.pi) * f

    pole = P1 * P2 + P3 * P4

    return pole

# [1] Fant, Gunnar, Johan Liljencrants, and Qi-guang Lin. "A four-parameter model of glottal flow." STL-QPSR 4.1985 (1985): 1-13.
# [2] Doval, Boris, Christophe d'Alessandro, and Nathalie Henrich. "The spectrum of glottal flow models." Acta acustica united with acustica 92.6 (2006): 1026-1046.
@nb.jit(nb.float64[:](nb.float64[:], nb.float64, nb.float64, nb.float64, nb.float64, nb.float64), nopython = True, cache = True)
def calcFlowDerivative(t, T0, Ee, tp, te, ta):
    """
    Calculate flow derivative for given lf model parameters

    Parameters
    ----------
    t: array_like
        List of time
    T0: float
        Fundamental period, 1 / f0
    Ee: float
        Pulse amplitude
    tp, te, ta: float, float, float
        LF model shape parameters, T0-normalized

    Returns
    ----------
    array_like
        Sample list, shape = t.shape
    """
    tp *= T0
    te *= T0
    ta *= T0
    wg, sinWgTe, cosWgTe, e, A, a, E0 = calcInternalParameter(T0, Ee, tp, te, ta)

    out = np.zeros(t.shape)
    o = t <= te
    c = np.logical_and(t <= T0, t > te)
    to = t[o]
    tc = t[c]
    out[o] = E0 * np.exp(a * to) * np.sin(wg * to)
    out[c] = -Ee * (np.exp((te - tc) * e) - np.exp((te - T0) * e)) / (e * ta) # [2] p.18., formula in [1] p.8 is wrong.
    return out

def calcVariantFlowDerivative(nSample, tList, TList, EeList, RdList, sr):
    # TODO: use antialias lf-model and use a proper window function
    """
    Calculate variant flow derivative form given lf model parameters

    Parameters
    ----------
    nSample: int
        Sample count of output
    tList: array_like
        List of sample position
    TList: array_like
        Fundamental period list at each time point, 1 / f0
    EeList: array_like
        Pulse amplitude at each time point
    RdList: float
        LF model shape parameter at each time point
    sr: int
        Sample rate

    Returns
    ----------
    array_like
        Sample list, shape = (nSample,)
    """
    nSample = nSample * 4
    tList = np.asarray(tList) * 4
    sr = sr * 4

    tList = np.concatenate(([min(-1, tList[0] - 1)], tList, [max(nSample, tList[-1] + 1)]))
    TList = np.concatenate(([TList[0]], TList, [TList[-1]]))
    EeList = np.concatenate(([EeList[0]], EeList, [EeList[-1]]))
    RdList = np.concatenate(([RdList[0]], RdList, [RdList[-1]]))
    yList = np.array([TList, EeList, RdList], dtype = np.float64)
    f = ipl.interp1d(tList, yList, kind="linear", copy = False)

    out = np.zeros(nSample)
    t = 0
    while t < nSample:
        T, Ee, Rd = f(t)
        dT = int(round(T * sr))
        t_list = np.linspace(0, T, dT)
        x = calcFlowDerivative(t_list, T, Ee, *calcParameterFromRd(Rd))
        ob, oe, ib, ie = getFrameRange(nSample, t, dT)
        out[ib:ie] = x[ob:oe]
        t += dT
    return sp.resample_poly(out, 1, 4)

def calcRdFromTenseness(tenseness):
    """
    Calculate Rd from given tenseness

    Parameters
    ----------
    tenseness: float
        In range (0.0, 1.0]

    Returns
    ----------
    float
        Rd
    """
    return max(1e-2, min(3 * (1.0 - tenseness), 3.0))

def calcTensenessFromRd(Rd):
    """
    Calculate tenseness form given Rd

    Parameters
    ----------
    Rd: float
        In range (0.0, 3.0]

    Returns
    ----------
    float
        tenseness
    """
    return max(0.0, min(1 - Rd / 3, 1.0 - 2.99 / 3))
