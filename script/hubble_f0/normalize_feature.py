import sys
sys.path.append("../../")

import numpy as np
import pylab as pl
from revoice import *
from revoice.common import *
import os, pickle

from config import *

feature_path = os.path.abspath("C:/train/f0detect_train_load.mmap")
n_hop_per_sample = 512

print("* Mapping...")
data = np.memmap(feature_path, dtype=np.float32, mode="r", order="C")
n_sample = data.shape[0] // (n_hop_per_sample * (n_feature + 1))
data = data.reshape(n_sample, n_hop_per_sample, n_feature + 1)

mean = np.zeros(n_feature, dtype=np.float64)
stdev = np.zeros(n_feature, dtype=np.float64)

pdata = data[:, :, 1:].reshape(n_sample * n_hop_per_sample, n_feature)
print("* Mean")
mean[:] = np.mean(pdata, axis=0)
print("* Stdev")
for i in range(n_sample * n_hop_per_sample):
  stdev += (pdata[i] - mean) ** 2
stdev /= n_sample * n_hop_per_sample
stdev = np.sqrt(stdev)

mean = mean.astype(np.float32)
stdev = stdev.astype(np.float32)
with open("normalize.pickle", "wb") as f:
  pickle.dump((mean, stdev), f)

pl.plot(mean)
pl.plot(stdev)
pl.show()