from .common import *
from . import sparsehmm
import tensorflow as tf

def f0CostFunction(y_true, y_pred):
  return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred, labels=y_true))

def vuvCostFunction(y_true, y_pred):
  return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=y_true))

class WeightClip(tf.keras.constraints.Constraint):
  '''Clips the weights incident to each hidden unit to be inside a range'''
  def __init__(self, c=0.998):
    self.c = c

  def __call__(self, p):
    return tf.clip_by_value(p, -self.c, self.c)

  def get_config(self):
    return {
      'name': self.__class__.__name__,
      'c': self.c
    }

def hubbleF0Network(inputData, isTrain=True, reuse=False, scopeName="net_hubble_f0"):
  wInit = tf.random_normal_initializer(stddev = 0.02)
  gInit = tf.random_normal_initializer(mean = 1, stddev = 0.02)
  constraint = WeightClip(0.998)

  Dense = tf.keras.layers.Dense
  GRU = tf.keras.layers.GRU
  Bidirectional = tf.keras.layers.Bidirectional
  tanh = tf.nn.tanh
  sigmoid = tf.nn.sigmoid
  softmax = tf.nn.softmax
  lrelu = tf.nn.leaky_relu

  with tf.variable_scope(scopeName, reuse=reuse):
    logits = Dense(64, activation=None, kernel_constraint=constraint, bias_constraint=constraint)(inputData)
    x = tf.nn.softmax(logits)
    return logits, x

def hubbleF0Network2(inputData, isTrain=True, reuse=False, scopeName="net_hubble_f0"):
  wInit = tf.random_normal_initializer(stddev = 0.02)
  gInit = tf.random_normal_initializer(mean = 1, stddev = 0.02)
  constraint = WeightClip(0.998)

  Dense = tf.keras.layers.Dense
  GRU = tf.keras.layers.GRU
  Bidirectional = tf.keras.layers.Bidirectional
  tanh = tf.nn.tanh
  sigmoid = tf.nn.sigmoid
  softmax = tf.nn.softmax
  lrelu = tf.nn.leaky_relu

  with tf.variable_scope(scopeName, reuse=reuse):
    x = Dense(128, activation=tanh, kernel_constraint=constraint, bias_constraint=constraint)(inputData)
    logits = Dense(65, activation=None, kernel_constraint=constraint, bias_constraint=constraint)(x)
    x = tf.nn.softmax(logits)
    return logits, x

def hubbleF0Network3(inputData, isTrain=True, reuse=False, scopeName="net_hubble_f0"):
  wInit = tf.random_normal_initializer(stddev = 0.02)
  gInit = tf.random_normal_initializer(mean = 1, stddev = 0.02)
  constraint = WeightClip(0.998)

  Dense = tf.keras.layers.Dense
  GRU = tf.keras.layers.GRU
  Bidirectional = tf.keras.layers.Bidirectional
  tanh = tf.nn.tanh
  sigmoid = tf.nn.sigmoid
  softmax = tf.nn.softmax
  lrelu = tf.nn.leaky_relu

  with tf.variable_scope(scopeName, reuse=reuse):
    x = Dense(128, activation=tanh, kernel_constraint=constraint, bias_constraint=constraint)(inputData)
    x = GRU(80, activation=tanh, recurrent_activation=sigmoid, return_sequences=True, kernel_constraint=constraint, bias_constraint=constraint, go_backwards=False)(x)
    logits = Dense(65, activation=None, kernel_constraint=constraint, bias_constraint=constraint)(x)
    x = tf.nn.softmax(logits)
    return logits, x

def hubbleVUVNetwork(inputData, isTrain=True, reuse=False, scopeName="net_hubble_vuv"):
  wInit = tf.random_normal_initializer(stddev = 0.02)
  gInit = tf.random_normal_initializer(mean = 1, stddev = 0.02)
  constraint = WeightClip(0.998)

  Dense = tf.keras.layers.Dense
  GRU = tf.keras.layers.GRU
  Bidirectional = tf.keras.layers.Bidirectional
  tanh = tf.nn.tanh
  sigmoid = tf.nn.sigmoid
  softmax = tf.nn.softmax
  lrelu = tf.nn.leaky_relu

  with tf.variable_scope(scopeName, reuse=reuse):
    x = Dense(32, activation=tanh, kernel_constraint=constraint, bias_constraint=constraint)(inputData)
    #x = Dense(72, activation=tanh, kernel_constraint=constraint, bias_constraint=constraint)(x)
    x = GRU(32, dropout=0.1, recurrent_dropout=0.1, activation=tanh, recurrent_activation=sigmoid, return_sequences=True, kernel_constraint=constraint, bias_constraint=constraint, go_backwards=False)(x)
    #b = GRU(72, dropout=0.1, recurrent_dropout=0.1, activation=tanh, recurrent_activation=sigmoid, return_sequences=True, kernel_constraint=constraint, bias_constraint=constraint, go_backwards=True)(x)
    #x = a + b
    logits = Dense(1, activation=None, kernel_constraint=constraint, bias_constraint=constraint)(x)
    x = tf.nn.sigmoid(logits)
    return logits, x

def hubbleVUVNetwork2(inputData, isTrain=True, reuse=False, scopeName="net_hubble_vuv"):
  wInit = tf.random_normal_initializer(stddev = 0.02)
  gInit = tf.random_normal_initializer(mean = 1, stddev = 0.02)
  constraint = WeightClip(0.998)

  Dense = tf.keras.layers.Dense
  GRU = tf.keras.layers.GRU
  Bidirectional = tf.keras.layers.Bidirectional
  tanh = tf.nn.tanh
  sigmoid = tf.nn.sigmoid
  softmax = tf.nn.softmax
  lrelu = tf.nn.leaky_relu

  with tf.variable_scope(scopeName, reuse=reuse):
    x = Dense(64, activation=tanh, kernel_constraint=constraint, bias_constraint=constraint)(inputData)
    x = Dense(32, activation=tanh, kernel_constraint=constraint, bias_constraint=constraint)(x)
    logits = Dense(1, activation=None, kernel_constraint=constraint, bias_constraint=constraint)(x)
    x = tf.nn.sigmoid(logits)
    return logits, x

_bandIdxList = np.arange(64, dtype=np.float32)
@nb.njit(fastmath=True)
def _f0TransProb(iFrm):
  o = 0.07869048**(np.abs(_bandIdxList - iFrm)**1.4368463)
  #o = np.zeros(64, dtype=np.float32)
  #o[:max(0, iFrm - 2)] = 0
  #o[min(iFrm + 3, 64):] = 0
  '''o[iFrm] = 1.0
  if iFrm >= 1:
    o[iFrm - 1] = 0.5
  if iFrm < 63:
    o[iFrm + 1] = 0.5
  if iFrm >= 2:
    o[iFrm - 2] = 0.25
  if iFrm < 62:
    o[iFrm + 2] = 0.25
  if iFrm >= 3:
    o[iFrm - 3] = 0.125
  if iFrm < 61:
    o[iFrm + 3] = 0.125'''
  #o[o < 1e-4] = 1e-4
  o /= np.sum(o)
  return o

class HubbleF0Trajectory:
  def __init__(self):
    self.model, self.initStateProb, self.sourceState, self.targetState, self.stateTransProb = self.createModel()

  def createModel(self):
    nBand = 64
    nState = nBand
    nTrans = nState * nState
    init = np.full(nState, 1.0 / nState, dtype=np.float32)
    frm = np.repeat(np.arange(nState, dtype=np.int32), nState)
    to = np.tile(np.arange(nState, dtype=np.int32), nState)
    transProb = np.zeros(nTrans, dtype=np.float32)
    for iFrm in range(nState):
      transProb[iFrm * nState:(iFrm + 1) * nState] = _f0TransProb(iFrm)

    return sparsehmm.ViterbiDecoder(nState, nTrans), init, frm, to, transProb

  def calcStateProb(self, obsProb):
    return obsProb / np.sum(obsProb)

  def __call__(self, obsProbList):
    (nHop, nBand) = obsProbList.shape
    assert nBand == 64, "Bad shape"

    # feed & decode
    self.model.initialize(self.calcStateProb(obsProbList[0]), self.initStateProb)
    self.model.preserve(nHop - 1)
    for iHop in range(1, nHop):
      self.model.feed(self.calcStateProb(obsProbList[iHop]), self.sourceState, self.targetState, self.stateTransProb)
    path = self.model.readDecodedPath()
    self.model.finalize()
    return path

class HubbleVUVTrajectory:
  def __init__(self):
    self.model, self.initStateProb, self.sourceState, self.targetState, self.stateTransProb = self.createModel()

  def createModel(self):
    nState = 2
    nTrans = nState * nState
    transSelf = 0.99
    transOther = 1.0 - transSelf
    init = np.full(nState, 1.0 / nState, dtype=np.float32)
    frm = np.repeat(np.arange(nState, dtype=np.int32), nState)
    to = np.tile(np.arange(nState, dtype=np.int32), nState)
    transProb = np.array([transSelf, transOther, transOther, transSelf], dtype=np.float32)

    return sparsehmm.ViterbiDecoder(nState, nTrans), init, frm, to, transProb

  def calcStateProb(self, obsProb):
    return np.array((1.0 - obsProb, obsProb), dtype=np.float32)

  def __call__(self, obsProbList):
    (nHop,) = obsProbList.shape

    # feed & decode
    self.model.initialize(self.calcStateProb(obsProbList[0]), self.initStateProb)
    self.model.preserve(nHop - 1)
    for iHop in range(1, nHop):
      self.model.feed(self.calcStateProb(obsProbList[iHop]), self.sourceState, self.targetState, self.stateTransProb)
    path = self.model.readDecodedPath()
    self.model.finalize()
    return path

class HubbleTracker:
  def __init__(self):
    self.model, self.initStateProb, self.sourceState, self.targetState, self.stateTransProb = self.createModel()

  def createModel(self):
    transSelf = 0.99
    transOther = 1.0 - transSelf
    nBand = 64
    nState = nBand + 1
    nTrans = nState * nState
    init = np.full(nState, 1.0 / nState, dtype=np.float32)
    frm = np.repeat(np.arange(nState, dtype=np.int32), nState)
    to = np.tile(np.arange(nState, dtype=np.int32), nState)
    transProb = np.zeros(nTrans, dtype=np.float32)
    for iFrm in range(nBand):
      p = _f0TransProb(iFrm)
      transProb[iFrm * nState:iFrm * nState + nBand] = p * transSelf
      transProb[iFrm * nState + nBand] = transOther
      transProb[iFrm * nState:iFrm * nState + nState] /= np.sum(p) + transOther
      '''import pylab as pl
      pl.plot(transProb[iFrm * nState:iFrm * nState + nState])
      pl.show()'''
      del p
    transProb[nBand * nState:nBand * nState + nBand] = transOther / nBand
    transProb[nBand * nState + nBand] = transSelf

    return sparsehmm.ViterbiDecoder(nState, nTrans), init, frm, to, transProb

  def calcStateProb(self, obsProb):
    return obsProb / np.sum(obsProb)

  def __call__(self, obsProbList):
    (nHop, nBand) = obsProbList.shape
    assert nBand == 65, "Bad shape"

    # feed & decode
    self.model.initialize(self.calcStateProb(obsProbList[0]), self.initStateProb)
    self.model.preserve(nHop - 1)
    for iHop in range(1, nHop):
      self.model.feed(self.calcStateProb(obsProbList[iHop]), self.sourceState, self.targetState, self.stateTransProb)
    path = self.model.readDecodedPath()
    self.model.finalize()
    return path