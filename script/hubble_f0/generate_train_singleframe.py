import sys
sys.path.append("../../")

import numpy as np
import pylab as pl
from revoice import *
from revoice.common import *
import gc, pickle

from config import *

mvf = sr * 0.5
noise_min, noise_max = noise_ratio_range
har_min, har_max = harmonic_range
amp_min, amp_max = amp_range

print("* Creating...")
out = np.memmap(train_singleframe_data_path, dtype=np.float32, mode="w+", shape=(n_sample_singleframe, n_hop_per_sample, n_feature + 1), order="C") # F0 + n_band * (SNR * 3 + IF * 2)

print("* Generating...")
@nb.njit(parallel=True, fastmath=True, cache=True)
def gen_if(x, out, i_freq, i_sample, h1, hd1, h2, hd2):
  (nh,) = h1.shape
  for i_hop in nb.prange(n_hop_per_sample):
    i_center = int(round(i_hop * hop_size))
    frame = getFrame(x, i_center, nh)
    out[i_sample, i_hop, 4 + i_freq * 5] = yang.calcYangIF(frame, h1, hd1)
    out[i_sample, i_hop, 5 + i_freq * 5] = yang.calcYangIF(frame, h2, hd2)

def gen_feature(x, out, i_sample):
  idx_list = (np.arange(0, n_hop_per_sample, dtype=np.float64) * hop_size).round().astype(np.int32)
  for i_freq, freq in enumerate(band_freq_list):
    rel_freq = freq / sr
    w = yang.createYangSNRParameter(rel_freq)
    out[i_sample, :, 1 + i_freq * 5] = np.log(yang.calcYangSNR(x, rel_freq * 0.5, w)[idx_list] + 1e-5)
    out[i_sample, :, 2 + i_freq * 5] = np.log(yang.calcYangSNR(x, rel_freq, w)[idx_list] + 1e-5)
    out[i_sample, :, 3 + i_freq * 5] = np.log(yang.calcYangSNR(x, rel_freq * 2, w)[idx_list] + 1e-5)
    del w

    h1, hd1 = yang.createYangIFParameter(rel_freq, rel_freq)
    h2, hd2 = yang.createYangIFParameter(rel_freq * 2, rel_freq)
    gen_if(x, out, i_freq, i_sample, h1, hd1, h2, hd2)

n_sample = getNSample(n_hop_per_sample, hop_size)
t = np.arange(-n_sample // 2, -n_sample // 2 + n_sample, dtype=np.float32) / sr
b, a = dc_iir_filter(50.0 / sr / 2)
for i_sample in range(0, n_sample_singleframe_ideal):
  print("  Ideal Sample %d/%d" % (i_sample, n_sample_singleframe_ideal))
  #w, wsr = loadWav("../../voices/yuri_orig.wav")
  #w = sp.resample_poly(w, sr, wsr)

  f0 = barkToFreq(np.random.uniform(band_start, band_stop))
  nHar = int(mvf / f0)
  hFreq = np.arange(1, nHar + 1, dtype=np.float32) * f0
  hAmp = np.zeros(nHar, dtype=np.float32)
  hAmp[0] = 1.0
  hAmp[1:] = np.random.uniform(har_min, har_max, nHar - 1)
  hAmp *= np.abs(lfmodel.calcSpectrum(hFreq, 1 / f0, 1.0, *lfmodel.calcParameterFromRd(np.random.uniform(0.3, 2.7))))
  hPhase = np.random.uniform(-np.pi, np.pi, nHar).astype(np.float32)
  x = hnm.synthSinusoid(t, hFreq, hAmp, hPhase, sr)
  x /= np.sqrt(np.mean(x * x))
  noise = np.random.uniform(-1.0, 1.0, n_sample)
  noise *= np.random.uniform(noise_min, noise_max) / np.sqrt(np.mean(noise * noise))
  x += noise
  del noise
  x /= np.max(np.abs(x))
  x *= np.random.uniform(amp_min, amp_max)
  #x = w[100:100+t.shape[0]]
  x = sp.filtfilt(b, a, x).astype(np.float32)

  out[i_sample, :, 0] = f0
  gen_feature(x, out, i_sample)
  '''pl.figure()
  pl.imshow(out[i_sample, :, 2::5].T, interpolation='nearest', aspect='auto', origin='lower', cmap="plasma")
  pl.figure()
  pl.imshow(out[i_sample, :, 4::5].T, interpolation='nearest', aspect='auto', origin='lower', cmap="plasma", vmin=band_freq_list[0] / sr, vmax=band_freq_list[-1] / sr)
  pl.show()'''