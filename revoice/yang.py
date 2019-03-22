import numpy as np
from .common import *
import pylab as pl
from . import accelerator

# https://arxiv.org/pdf/1605.07809.pdf
# Heavy copy from https://github.com/Sleepwalking/ciglet/

#window_coeff = np.array([0.338946, 0.481973, 0.161054, 0.018027], dtype=np.float32) # nuttall83
window_coeff = np.array([0.3635819, 0.4891775, 0.1365995, 0.0106411], dtype=np.float32) # nuttall98
#window_coeff = np.array([0.5, 0.5, 0, 0], dtype=np.float32) # hanning

def x2as_fast(x, h, ws, mt):
  (nw,) = h.shape
  (nx,) = x.shape
  nw_2 = nw // 2

  y1 = np.zeros(nx + nw - 1, dtype=np.complex64)
  y2 = np.zeros(nx + nw - 1, dtype=np.complex64)

  #y1 = sp.fftconvolve(x, h)[nw_2:-nw_2 + 1]
  if mt:
    accelerator.segconvRealCplx(x, h, y1)
  else:
    accelerator.segconvRealCplxSt(x, h, y1)
  y1 = y1[nw_2:-nw_2 + 1]
  y1 /= np.abs(y1) + 1e-8
  #y2 = sp.fftconvolve(y1, h)[nw_2:-nw_2 + 1]
  if mt:
    accelerator.segconvCplx(y1, h, y2)
  else :
    accelerator.segconvCplxSt(y1, h, y2)
  y2 = y2[nw_2:-nw_2 + 1]

  # r = y1 - y2
  y1 -= y2
  r = y1
  del y1, y2
  a = np.abs(r, dtype=np.float32)
  del r
  a *= a

  s = np.zeros(nx + nw - 1, dtype=np.float32)
  #s = sp.fftconvolve(a, ws)[nw_2:-nw_2 + 1]
  if mt:
    accelerator.segconv(a, ws, s)
  else:
    accelerator.segconvSt(a, ws, s)
  s = s[nw_2:-nw_2 + 1]
  del a, ws
  s[:nw] = s[nw - 1]
  s[-nw:] = s[-nw]
  return s

def x2as(x, h, ws):
  (nw,) = h.shape
  (nx,) = x.shape
  nw_2 = nw // 2

  y1 = sp.fftconvolve(x, h, mode="full")[nw_2:-nw_2 + 1] # not equal to "same"
  y1 = y1
  y1 /= np.abs(y1) + 1e-8
  y2 = sp.fftconvolve(y1, h, mode="full")[nw_2:-nw_2 + 1]

  # r = y1 - y2
  y1 -= y2
  r = y1
  del y1, y2
  a = np.abs(r, dtype=np.float32)
  del r
  a *= a
  
  s = sp.fftconvolve(a, ws, mode="full")[nw_2:-nw_2 + 1]
  del a, ws
  s[:nw] = s[nw - 1]
  s[-nw:] = s[-nw]
  return s


def x2as_sf(x, h, ws):
  (nw,) = h.shape
  (nx,) = x.shape
  nw_2 = nw // 2

  y1 = sp.fftconvolve(x, h, mode="full")[nw:-nw + 2]
  y1 /= np.abs(y1) + 1e-8
  y2 = sp.fftconvolve(y1, h, mode="full")[nw:-nw + 2]
  
  r = y1[nw_2:-nw_2 + 1] - y2
  del y1, y2
  a = np.abs(r, dtype=np.float32)
  del r
  a *= a
  
  s = sp.fftconvolve(a, ws, mode="full")[nw:-nw + 2]
  del a, ws
  assert s.size == 1, "Bad assumption"
  return s[0]

def x2as_sf_fast(x, h, ws):
  (nw,) = h.shape
  (nx,) = x.shape

  y1 = np.zeros(nx + nw - 1, dtype=np.complex64)
  
  accelerator.segconvRealCplxSt(x, h, y1)
  y1 = y1[nw:-nw + 2]
  y1 /= np.abs(y1) + 1e-8

  y2 = np.zeros(y1.size + nw - 1, dtype=np.complex64)
  accelerator.segconvCplxSt(y1, h, y2)
  y2 = y2[nw:-nw + 2]
  
  r = y1[nw // 2:-nw // 2 + 1] - y2
  del y1, y2
  a = np.abs(r, dtype=np.float32)
  del r
  a *= a
  
  s = np.zeros(a.size + nw - 1, dtype=np.float32)
  accelerator.segconvSt(a, ws, s)
  s = s[nw:-nw + 2]
  del a, ws
  assert s.size == 1, "Bad assumption"
  return s[0]

@nb.njit()
def createYangSNRParameter(bw):
  nw = int(np.ceil(4.0 / bw))
  if nw % 2 != 0:
    nw += 1
  nw_2 = nw // 2.0
  omegaw = 2.0 * np.pi * bw / 4.0
  w = np.zeros(nw, dtype=np.float32)
  for iCoeff in range(4):
    jw = np.cos(iCoeff * omegaw * (np.arange(0, nw).astype(np.float32) - nw_2))
    w += window_coeff[iCoeff] * jw
  w /= np.sum(w)
  return w

def calcYangSNRSingleFrame(x, f, w):
  (nw,) = w.shape
  assert nw % 2 == 0
  (nx,) = x.shape
  assert nx == 1 + (nw - 1) * 3
  nw_2 = nw // 2.0

  omegah = 2 * np.pi * f
  h = w * np.exp(1j * omegah * (np.arange(0, nw).astype(np.float32) - nw_2))
  return x2as_sf_fast(x, h, w)

def calcYangSNR(x, f, w, mt=True):
  (nw,) = w.shape
  assert nw % 2 == 0
  nw_2 = nw // 2.0

  omegah = 2 * np.pi * f
  h = w * np.exp(1j * omegah * (np.arange(0, nw).astype(np.float32) - nw_2))
  return x2as_fast(x, h, w, mt)

@nb.njit(fastmath=True)
def createYangIFParameter(f, bw):
  nh = int(np.ceil(4.0 / bw))
  if nh % 2 != 0:
    nh += 1
  omega = 2.0 * np.pi * f
  omega_bw = 2.0 * np.pi * bw

  h = np.zeros(nh, dtype=np.complex64)
  hd = np.zeros(nh, dtype=np.complex64)
  for i in range(nh):
    for k in range(4):
      h.real[i] += window_coeff[k] * np.cos(k * omega_bw * (i - nh / 2))
      hd.real[i] += -omega_bw * k * window_coeff[k] * np.sin(k * omega_bw * (i - nh / 2))
  for i in range(nh):
    sini = np.sin(omega * (i - nh / 2))
    cosi = np.cos(omega * (i - nh / 2))
    w = h.real[i]
    wd = hd.real[i]
    h[i] = (w * cosi) + (w * sini) * 1j
    hd[i] = (wd * cosi - omega * w * sini) + (omega * w * cosi + wd * sini) * 1j
  
  return h, hd

@nb.njit(fastmath=True)
def calcYangIF(x, h, hd):
  assert x.shape == h.shape == hd.shape
  (nh,) = h.shape
  assert nh % 2 == 0
  
  y = np.sum(h * x)
  yd = np.sum(hd * x)
  return (y.real * yd.imag - y.imag * yd.real) / (y.real * y.real + y.imag * y.imag + 1e-8) / 2.0 / np.pi
