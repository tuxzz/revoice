from .common import *
from . import accelerator
from . import yang
from . import energy
from . import sparsehmm

def loadNet():
  import os, pickle
  with open(os.path.join(os.path.dirname(__file__), "hubble_f0_vuv_sf.pickle"), "rb") as f:
    net = pickle.load(f)
  return net["mean"], net["stdev"], net["net_hubble_vuv/dense/kernel:0"], net["net_hubble_vuv/dense/bias:0"], net["net_hubble_vuv/dense_1/kernel:0"], net["net_hubble_vuv/dense_1/bias:0"]

workSr = 4000
_n_band = 64
_bark_start = freqToBark(66.0)
_bark_stop = freqToBark(1100.0)
_band_freq_list = barkToFreq(np.linspace(_bark_start, _bark_stop, _n_band, dtype=np.float32))

_mean, _stdev, _k0, _b0, _k1, _b1 = loadNet()
assert _k0.shape == (192, 128)
assert _b0.shape == (128,)
assert _k1.shape == (128, _n_band + 1)
assert _b1.shape == (_n_band + 1,)

@nb.njit(fastmath=True, cache=True)
def _gen_if(x, out, i_freq, h1, hd1, h2, hd2, hop_size, n_hop):
  (nh,) = h1.shape
  for i_hop in nb.prange(n_hop):
    i_center = int(round(i_hop * hop_size))
    frame = getFrame(x, i_center, nh)
    out[i_hop, i_freq, 2] = yang.calcYangIF(frame, h1, hd1)
def _generate_sample_1(x: np.ndarray, n_hop: int, hop_size: float, mt: bool = False) -> np.ndarray:
  (n_x,) = x.shape
  out = np.zeros((n_hop, _n_band, 3), dtype=np.float32)
  idx_list = (np.arange(0, n_hop, dtype=np.float64) * hop_size).round().astype(np.int32)
  for (i_freq, freq) in enumerate(_band_freq_list):
    rel_freq = freq / workSr
    w = yang.createYangSNRParameter(rel_freq)
    out[:, i_freq, 0] = np.log(yang.calcYangSNR(x, rel_freq * 0.5, w, mt=mt)[idx_list] + 1e-5)
    out[:, i_freq, 1] = np.log(yang.calcYangSNR(x, rel_freq, w, mt=mt)[idx_list] + 1e-5)
    del w

    h1, hd1 = yang.createYangIFParameter(rel_freq, rel_freq)
    h2, hd2 = yang.createYangIFParameter(rel_freq * 2, rel_freq)
    _gen_if(x, out, i_freq, h1, hd1, h2, hd2, hop_size, n_hop)
  return out

_band_idx_list = np.arange(64, dtype=np.float32)
@nb.njit(fastmath=True)
def _f0_trans_prob(iFrm):
  o = 0.07869048**(np.abs(_band_idx_list - iFrm)**1.4368463)
  o /= np.sum(o)
  return o

@nb.njit(fastmath=True, cache=True)
def _run_net(feature_list):
  (_n_hop, n_feature) = feature_list.shape
  assert n_feature == _n_band * 3, "Bad shape"

  feature_list = feature_list.copy()
  feature_list -= _mean.reshape(1, _n_band * 3)
  feature_list /= _stdev.reshape(1, _n_band * 3)

  l0 = np.dot(feature_list, _k0)
  l0 += _b0
  l0 = np.tanh(l0)
  l1 = np.dot(l0, _k1)
  l1 += _b1
  exp_l1 = np.exp(l1)
  exp_l1 /= np.sum(exp_l1, axis=1)
  return exp_l1

def generateMap(x, hopSize):
  n_hop = getNFrame(x.size, hopSize)
  feature_list = _generate_sample_1(x, n_hop, hopSize, mt=True)
  prob_map = _run_net(feature_list.reshape(n_hop, _n_band * 3))
  return prob_map

def _snr(relFreq, x, iCenter):
  w = yang.createYangSNRParameter(relFreq)
  frame = getFrame(x, iCenter, 1 + (w.size - 1) * 3)

  return yang.calcYangSNRSingleFrame(frame, relFreq, w)

#@nb.njit(fastmath=True)
def refine(x, freqList, hopSize, maxIter=16, maxMove=10.0):
  (nHop,) = freqList.shape
  out = np.zeros(nHop, dtype=np.float32)
  for (iHop, f) in enumerate(freqList):
    if f > 0.0:
      iCenter = int(round(iHop * hopSize))
      binIdx = freqToBin(f)
      minBinIdx = max(0, binIdx - 2)
      maxBinIdx = min(binIdx + 3, _n_band)
      minBinFreq = int(_band_freq_list[minBinIdx])
      maxBinFreq = int(_band_freq_list[maxBinIdx])
      f = minimizeScalar(lambda v:_snr(v, x, iCenter), np.linspace(minBinFreq / workSr, maxBinFreq / workSr, 32)) * workSr
    out[iHop] = f
  return out

@nb.vectorize()
def binToFreq(binIdx):
  if binIdx == 64:
    return 0.0
  else:
    return _band_freq_list[binIdx]

@nb.vectorize()
def freqToBin(freq):
  if freq <= 0.0:
    return 64
  else:
    return np.argmin(np.abs(_band_freq_list - freq))

class Tracker:
  def __init__(self, **kwargs):
    self.hopSize = kwargs.get("hopSize", workSr * 0.0025)
    self.model, self.initStateProb, self.sourceState, self.targetState, self.stateTransProb = self.createModel()

  def createModel(self):
    n_band = _n_band
    trans_self = 0.99
    trans_other = 1.0 - trans_self
    n_state = n_band + 1
    n_trans = n_state * n_state
    init = np.full(n_state, 1.0 / n_state, dtype=np.float32)
    frm = np.repeat(np.arange(n_state, dtype=np.int32), n_state)
    to = np.tile(np.arange(n_state, dtype=np.int32), n_state)
    trans_prob = np.zeros(n_trans, dtype=np.float32)
    for iFrm in range(n_band):
      p = _f0_trans_prob(iFrm)
      p *= trans_self
      trans_prob[iFrm * n_state:iFrm * n_state + n_band] = p
      trans_prob[iFrm * n_state + n_band] = trans_other
      del p
    trans_prob[n_band * n_state:n_band * n_state + n_band] = trans_other / n_band
    trans_prob[n_band * n_state + n_band] = trans_self

    return sparsehmm.ViterbiDecoder(n_state, n_trans), init, frm, to, trans_prob

  def __call__(self, obsProbList):
    (_n_hop, n_band) = obsProbList.shape
    assert n_band == 65, "Bad shape"

    # feed & decode
    self.model.initialize(obsProbList[0], self.initStateProb)
    self.model.feed(obsProbList[1:], self.sourceState, self.targetState, self.stateTransProb)
    path = self.model.readDecodedPath()
    self.model.finalize()
    return path

class F0Tracker:
  def __init__(self, **kwargs):
    self.hopSize = kwargs.get("hopSize", workSr * 0.0025)
    self.model, self.initStateProb, self.sourceState, self.targetState, self.stateTransProb = self.createModel()

  def createModel(self):
    n_band = _n_band
    n_state = n_band
    n_trans = n_state * n_state
    init = np.full(n_state, 1.0 / n_state, dtype=np.float32)
    frm = np.repeat(np.arange(n_state, dtype=np.int32), n_state)
    to = np.tile(np.arange(n_state, dtype=np.int32), n_state)
    trans_prob = np.zeros(n_trans, dtype=np.float32)
    for iFrm in range(n_band):
      trans_prob[iFrm * n_state:iFrm * n_state + n_state] = _f0_trans_prob(iFrm)

    return sparsehmm.ViterbiDecoder(n_state, n_trans), init, frm, to, trans_prob

  def calcStateProb(self, obsProb):
    return obsProb[:64] / np.sum(obsProb[:64])
  
  def __call__(self, obsProbList):
    (n_hop, n_band) = obsProbList.shape
    assert n_band == 65, "Bad shape"

    # feed & decode
    self.model.initialize(self.calcStateProb(obsProbList[0]), self.initStateProb)
    self.model.preserve(n_hop - 1)
    for i_hop in range(1, n_hop):
      self.model.feed(self.calcStateProb(obsProbList[i_hop]), self.sourceState, self.targetState, self.stateTransProb)
    path = self.model.readDecodedPath()
    self.model.finalize()
    return path