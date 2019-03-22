import numpy as np
import scipy.optimize as so
from .common import *

# [1] Huber, Stefan, and Axel Roebel. "On the use of voice descriptors for glottal source shape parameter estimation." Computer Speech & Language 28.5 (2014): 1170-1194.
# [2] Fant, Gunnar. "The LF-model revisited. Transformations and frequency domain analysis." Speech Trans. Lab. Q. Rep., Royal Inst. of Tech. Stockholm 2.3 (1995): 40.
@nb.jit(nb.types.Tuple((nb.float32, nb.float32, nb.float32))(nb.float32), fastmath=True, nopython=True, cache=True)
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
@nb.jit(nb.types.Tuple((nb.float32, nb.float32, nb.float32, nb.float32, nb.float32, nb.float32, nb.float32))(nb.float32, nb.float32, nb.float32, nb.float32, nb.float32), fastmath=False, nopython=True, cache=True)
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
    assert T0 > 0.0 and Ee >= 0.0 and tp > 0.0 and te > 0.0 and ta > 0.0
    wg = np.pi / tp #[1](2)
    sinWgTe = np.sin(wg * te)
    cosWgTe = np.cos(wg * te)

    e = max(1.0, (-ta * lambertW((te - T0) * np.exp((te - T0) / ta) / ta) + te - T0) / (ta * te - ta * T0)) # [1] (12)
    if np.isnan(e) or np.isinf(e) or e > 2.0 / (ta + 1e-9):
        print("[WARNING] Failed to solve lf parameter `e`, got e =", e)
        e = 1.0
    
    eTeT0 = np.exp(e * (te - T0))
    #A = e * ta / (e * e * ta) + (te - T0) * (1.0 - e * ta) / (e * ta)
    A = (1.0 - eTeT0) / (e * e * ta) + (te - T0) * eTeT0 / (e * ta); # [3] p.18, integral{0, T0} Ug(t) dt

    a = wg * (A * sinWgTe * wg - cosWgTe + 1)
    for _ in range(8):
        a = a - ((a * a + wg * wg) * sinWgTe * A + wg * (np.exp(-a * te) - cosWgTe) + a * sinWgTe) / (sinWgTe * (2 * A * a + 1) - wg * te * np.exp(-te * a))
    if np.isnan(a) or np.isinf(a) or a >= 1e9:
        print("[WARNING] Failed to solve lf parameter `a`, got a =", a)
        a = 0.0
    
    '''afunc = lambda x : (x * x + wg * wg) * sinWgTe * A + wg * (np.exp(-x * te) - cosWgTe) + x * sinWgTe
    dafunc = lambda x : sinWgTe * (2 * A * x + 1) - wg * te * np.exp(-te * x)
    aa = so.fsolve(afunc, 0.0, fprime = dafunc)[0]
    print(a, aa, te)
    assert np.abs((a - aa) / aa) < 1e-8, "Bad Precison"'''
    
    assert a < 1e9, "Failed to solve lf parameter"

    E0 = -Ee / (np.exp(a * te) * sinWgTe) # (5)
    return wg, sinWgTe, cosWgTe, e, A, a, E0

# [1] Doval, Boris, Christophe d'Alessandro, and Nathalie Henrich. "The spectrum of glottal flow models." Acta acustica united with acustica 92.6 (2006): 1026-1046.
@nb.jit(nb.complex64[:](nb.float32[:], nb.float32, nb.float32, nb.float32, nb.float32, nb.float32), fastmath=True, nopython=True, cache=True)
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
    assert (f > 0.0).all(), "f must be greater than 0"
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

    return pole.astype(np.complex64)

def antiAliasedCExponentialSegment(bet, tt, two):
    tw = two * 1.5
    alph = np.pi / tw
    aa = np.array([0.2624710164, 0.4265335164, 0.2250165621, 0.0726831633, 0.0125124215, 0.0007833203], dtype=np.float64)
    outbuf = np.zeros_like(tt)
    for k in range(len(aa)):
        seg1 = tt[(tt >= tw) & (tt <= tw+1)]
        seg2 = tt[np.abs(tt) < tw]
        seg3 = tt[np.abs(1-tt) <= tw]
        y1 = (k*alph*np.sin(k*alph*seg2)-bet*np.cos(k*alph*seg2) +(-1)**k*bet*np.exp(bet*tw)*np.exp(bet*seg2))/(k**2 * alph**2 + bet*bet)
        yy = ((-1)**k * bet * np.exp(bet * seg1) * (np.exp(bet * tw) - np.exp(-bet * tw))/(k**2 * alph**2 + bet*bet))
        y3 = ((k*alph*np.exp(bet)*np.sin(k*alph*(seg3-1)) + bet*np.exp(bet*(tw+seg3))*(-1)**k - bet*np.exp(bet)*np.cos(k*alph*(seg3-1))) / (k**2 * alph**2 + bet*bet))
        outbuf[(tt >= tw) & (tt <= tw+1)] += aa[k] * yy
        outbuf[np.abs(tt) < tw] += aa[k] * y1
        outbuf[np.abs(1-tt) <= tw] -= aa[k] * y3
    return outbuf / (tw * aa[0] * 2)

def antiAliasedPolynomialSegmentR(poly_coeff, tt, two):
    tw = two * 1.5
    hc = np.array([0.2625000000, 0.4265625000, 0.2250000000, 0.0726562500, 0.0125000000, 0.0007812500], dtype=np.float64)
    m = len(hc) - 1
    hc = hc / (2 * tw * hc[0])
    n = len(poly_coeff) - 1

    C = np.zeros((n + 1, m), dtype=np.float64)
    S = np.zeros((n + 1, m), dtype=np.float64)
    U = np.zeros((n + 1, 1 + n + 1), dtype=np.float64)
    V = np.zeros((n + 1, n + 1), dtype=np.float64)
    r = 0
    for k in range(1, m + 1):
        C[r, k - 1] = 0.0
        S[r, k - 1] = tw / (k * np.pi) * hc[k]
    U[r, 1] = hc[0]
    U[r, 0] = -U[r, 1:] * ((-tw) ** np.arange(1, n + 2).reshape(n + 2 - 1, 1))
    V[r, 0] = 1
    for r in range(1, n + 1):
        for k in range(1, m + 1):
            C[r, k - 1] = -((r * tw) / (k * np.pi)) * S[r - 1, k - 1]
            S[r, k - 1] = ((r * tw) / (k * np.pi)) * C[r - 1, k - 1]
        for k in range(1, n + 2):
            U[r - 1, k - 1] = (r / k) * U[r - 1, k - 1]
        U[r, 0] = -(U[r, 1:] * ((-tw) ** np.arange(1, n + 2)) + C[r, :] * ((-1) ** np.arange(1, m + 1)))
        for k in range(1, n + 1):
            V[r, k] = (r / k) * V[r - 1, k - 1]
        V[r, 0] = C[r, :] * ((-1) ** np.arange(1, m + 1)) + U[r, :] * ((tw) ** np.arange(0, n + 2)) - V[r, 1:] * ((tw) ** np.arange(1, n + 1))
    
    B = np.zeros((n + 1, n + 1), dtype=np.float32)
    for ii in range(1, n + 2):
        for jj in range(1, ii + 1):
            nr = ii - 1
            nk = jj - 1
            B[ii - 1, jj - 1] = np.math.factorial(nr) / np.math.factorial(nk) / np.math.factorial(nr - nk)
    c0 = poly_coeff * C
    s0 = poly_coeff * S
    u0 = poly_coeff * U
    v = poly_coeff * V
    c1 = poly_coeff * B * C
    s1 = poly_coeff * B * S
    u1 = poly_coeff * B * U

    output = tt * 0
    tt1 = tt[(tt > -tw) & (tt <= tw)]
    tt2 = tt[(tt > tw) & (tt <= 1 - tw)]
    tt3 = tt[(tt > 1 - tw) & (tt <= 1 + tw)]
    dt3 = tt[(tt > 1 - tw) & (tt <= 1 + tw)] - 1
    tt4 = tt[(tt > tw) & (tt <= 1 + tw)]

    tm1 = tt1 ** np.arange(0, n + 2)
    tm2 = tt2 ** np.arange(0, n + 2)
    tm3 = tt3 ** np.arange(0, n + 2)
    dm3 = dt3 ** np.arange(0, n + 2)
    tm4 = tt4 ** np.arange(0, n + 2)

    tcs1 = tt1 * np.arange(1, m + 1)
    dcs3 = dt3 * np.arange(1, m + 1)

    g1 = np.cos(np.pi * tcs1 / tw) * c0 + np.sin(np.pi * tcs1 / tw) * s0 + tm1 * u0
    g2 = tm2[:, 0:n + 1] * v
    if tw > 1 - tw:
        g3 = -(np.cos(np.pi * dcs3 / tw) * c1 + np.sin(np.pi * dcs3 / tw) * s1 + dm3 * u1)
    else:
        g3 = tm3[:, 0:n + 1] * v - (np.cos(np.pi * dcs3 / tw) * c1 + np.sin(np.pi * dcs3 / tw) * s1 + dm3 * u1)
    output[(tt > -tw) & (tt <= tw)] += g1
    output[(tt > tw) & (tt <= 1 - tw)] += g2
    output[(tt > 1 - tw) & (tt <= 1 + tw)] += g3
    if tw > 1 - tw:
        output[(tt > tw) & (tt <= 1 + tw)] += tm4[:, 0:n + 1] * v
    return output

def calcAAFlowDerivative(t, T0, Ee, tp, te, ta, two):
    tp *= T0
    te *= T0
    ta *= T0
    wg, _, _, e, _, a, E0 = calcInternalParameter(T0, Ee, tp, te, ta)

    out = np.zeros(t.shape, dtype=np.float32)
    sOpen = np.logical_and(t >= 0, t <= te)
    sClose = np.logical_and(t > te, t <= T0)
    tOpen = t[sOpen]
    tClose = t[sClose]
    cf = -np.exp(-a * te) / np.sin(wg * te)
    opening = E0 * np.exp(a * to) * np.sin(wg * to)
    closing = -Ee * (np.exp((te - tc) * e) - np.exp((te - T0) * e)) / (e * ta) # [2] p.18., formula in [1] p.8 is wrong.
    
    # AA
    margin = two * 2
    select1 = t >= -te & t <= te + margin
    select2 = t > te - margin & t <= T0 + margin
    exSeg1 = t[select1]
    exSeg2 = t[select2]
    piece1 = exSeg1 / te
    piece2 = (exSeg2 - te) / (T0 - te)
    tw1 = two / te
    tw2 = two / (T0 - te)
    beta1 = (a + 1j * wg) * te
    out1 = cf * antiAliasedCExponentialSegment(beta1, piece1, tw1).imag
    beta2 = -e * (T0 - te)
    tmp1 = antiAliasedCExponentialSegment(beta2, piece2, tw2).real
    tmp2 = antiAliasedPolynomialSegmentR(np.array([1.0, 0.0], dtype=np.float32), piece2, tw2)
    out2 = -tmp1 / (e * ta); 
    out3 = np.exp(-e * (T0 - te)) * tmp2 / (e * ta)
    opening = cf * np.exp(a * sOpen) * np.sin(wg * sOpen)
    closing = -1 / (e * ta) * (np.exp(-e * (sClose - te)) - np.exp(-e * (T0 - te)))
    out[select1] = out1
    out[select2] += out2
    out[select2] += out3

    return out

# [1] Fant, Gunnar, Johan Liljencrants, and Qi-guang Lin. "A four-parameter model of glottal flow." STL-QPSR 4.1985 (1985): 1-13.
# [2] Doval, Boris, Christophe d'Alessandro, and Nathalie Henrich. "The spectrum of glottal flow models." Acta acustica united with acustica 92.6 (2006): 1026-1046.
@nb.jit(nb.float32[:](nb.float32[:], nb.float32, nb.float32, nb.float32, nb.float32, nb.float32), fastmath=True, nopython=True, cache=True)
def calcFlowDerivative(t, T0, Ee, tp, te, ta):
    """
    Calculate flow derivative at each time `t` for given lf model parameters

    Parameters
    ----------
    t: array_like
        List of time
    T0: float
        Fundamental period, 1 / f0
    Ee: float
        Pulse amplitude
    tp, te, ta: float, float, floato
        LF model shape parameters, T0-normalized

    Returns
    ----------
    array_like
        Sample list, shape = t.shape
    """
    tp *= T0
    te *= T0
    ta *= T0
    wg, _, _, e, _, a, E0 = calcInternalParameter(T0, Ee, tp, te, ta)

    out = np.zeros(t.shape, dtype=np.float32)
    o = np.logical_and(t >= 0, t <= te)
    c = np.logical_and(t <= T0, t > te)
    to = t[o]
    tc = t[c]
    out[o] = E0 * np.exp(a * to) * np.sin(wg * to)
    out[c] = -Ee * (np.exp((te - tc) * e) - np.exp((te - T0) * e)) / (e * ta) # [2] p.18., formula in [1] p.8 is wrong.
    return out

@nb.jit(nb.float32[:](nb.float32[:], nb.float32, nb.float32, nb.float32, nb.float32, nb.float32), fastmath=True, nopython=True, cache=True)
def calcGlottalOpenness(t, T0, Ee, tp, te, ta):
    """
    Calculate glottal opennes at each time `t` for given lf model parameters

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

    out = np.zeros(t.shape, dtype=np.float32)
    o = t <= te
    c = np.logical_and(t <= T0, t > te)
    to = t[o]
    tc = t[c]
    out[o] = E0 * (np.exp(a * to) * (a * np.sin(wg * to) - wg * np.cos(wg * to)) + wg) / (a * a + wg * wg)
    vte = E0 * (np.exp(a * te) * (a * np.sin(wg * te) - wg * np.cos(wg * te)) + wg) / (a * a + wg * wg)
    u = (te - tc)
    T = (te - T0)
    out[c] = vte - Ee * (e * u * np.exp(e * T) - np.exp(e * u) + 1) / (ta * e * e)
    return out

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

@nb.jit(nb.types.Tuple((nb.float32, nb.float32, nb.float32))(nb.float32, nb.float32, nb.float32, nb.float32), fastmath=True, nopython=True, cache=True)
def calcGFModelParameterFromLFModel(T0, tp, te, ta):
    """
    Calculate Fa, Rk, Rg from T0, tp, te, ta

    Parameters
    ----------
    tp: float
    te: float
    ta: float

    Returns
    ----------
    (float, float, float)
        Fa, Rk, Rg
    """

    Fa = 1.0 / (ta * T0)
    Rk = (te - tp) / tp
    Rg = 0.5 / tp
    return (Fa, Rk, Rg)

@nb.jit(nb.types.Tuple((nb.float32, nb.float32, nb.float32))(nb.float32, nb.float32, nb.float32, nb.float32), fastmath=True, nopython=True, cache=True)
def calcLFModelParameterFromGFModel(T0, Fa, Rk, Rg):
    """
    Calculate tp, te, ta from T0, Fa, Rk, Rg

    Parameters
    ----------
    Fa: float
        Return phase frequency(Hz)
    Rk: float
        Decay duration relative to rising duration
    Rg: float
        Rising duration relative to period length

    Returns
    ----------
    (float, float, float)
        tp, te, ta
    """

    ta = 1 / (Fa * T0)
    tp = 0.5 / Rg
    te = tp + tp * Rk
    return (tp, te, ta)