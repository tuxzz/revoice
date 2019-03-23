import sys
sys.path.append("../../")

import numpy as np
import pylab as pl
from revoice import *
from revoice.common import *
import os, pickle

from config import *

f0_path = os.path.abspath("/home/tuxzz/vuvdetect_train_f0_marcob.mmap")
feature_path = os.path.abspath("/home/tuxzz/vuvdetect_train_feature_marcob.mmap")
base_band_idx = 0

n_hop_per_sample = 256

print("* Mapping...")
f0_list = np.memmap(f0_path, dtype=np.float32, mode="r", order="C")
n_sample = f0_list.size // n_hop_per_sample
f0_list = f0_list.reshape(n_sample, n_hop_per_sample)

#print("* Diff F0")
#band_list = np.argmin(np.abs(f0_list.reshape(n_sample, 1) - band_freq_list.reshape(1, n_band)), axis=1)

print("* Integrate diff..")
@nb.njit()
def worker(f0_list):
  diff_band_map = np.zeros(n_band + 1, dtype=np.float64)
  n = 0
  p = -1
  for f0 in f0_list:
    if f0 > 0.0:
      i_band = np.argmin(np.abs(f0 - band_freq_list))
    else:
      i_band = n_band
    if p == base_band_idx:
      diff_band_map[i_band] += 1.0
      n += 1
    p = i_band
  if n > 0:
    diff_band_map /= n
  return diff_band_map
diff_band_map = np.zeros(n_band + 1, dtype=np.float64)
for f0_sublist in f0_list:
  diff_band_map += worker(f0_sublist)
diff_band_map /= n_sample
print(diff_band_map)

fac = np.max(diff_band_map)
@nb.njit(fastmath=True)
def fev(para):
  o = ((np.abs(para[0])**(np.abs(np.arange(n_band) - base_band_idx)**para[1]))) * fac
  return o

@nb.njit(fastmath=True)
def loss(para):
  return np.mean(np.abs((np.sqrt(fev(para)) - np.sqrt(diff_band_map)))**2)
'''
res = so.basinhopping(loss, [0.5, 0.5], niter=100)
print(res)
para = np.abs(res.x)
'''
para = np.array([0.07869048,  1.4368463])
print("* Plot..")
pl.plot(diff_band_map)
#pl.plot(np.linspace(-diff_radius, diff_radius, n_diff_band), diff_f0_map)
'''pl.plot(fev(para))
v = fev(para)
v[v < 1e-3] = 1e-3
v /= np.sum(v)
pl.plot(v)'''
pl.show()