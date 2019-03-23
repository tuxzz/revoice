import sys
sys.path.append("../../")

from revoice.common import *

sr = 4000
n_band = 64
hop_size = sr * 0.0025
band_start = freqToBark(66.0)
band_stop = freqToBark(1100.0)
band_freq_list = barkToFreq(np.linspace(band_start, band_stop, n_band, dtype=np.float32))
n_feature = n_band * 5