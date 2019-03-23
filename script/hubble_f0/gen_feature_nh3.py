import numpy as np
import numba as nb
import os

from config import *

feature_path = os.path.abspath("C:/train/vuvdetect_train_feature_marcob.mmap")
feature_out_path = os.path.abspath("./vuvdetect_train_feature_marcob_nh3.mmap")

feature_data = np.memmap(feature_path, dtype=np.float32, mode="r", order="C")
n_sample = feature_data.shape[0] // n_feature
feature_data = feature_data.reshape(n_sample, n_band, 5)

out_data = np.memmap(feature_out_path, dtype=np.float32, mode="w+", order="C", shape=(n_sample, n_band, 3))
@nb.njit()
def worker(out_data):
  for i_sample in range(n_sample):
    out_data[i_sample, :, 0:2] = feature_data[i_sample, :, 0:2]
    out_data[i_sample, :, 2] = feature_data[i_sample, :, 3]
worker(out_data)