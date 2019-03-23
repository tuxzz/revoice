import sys
sys.path.append("../../")

import numpy as np
import pylab as pl
from revoice import *
from revoice.common import *

import os, pickle
import multiprocessing as mp
import win32process as wp

from config import *

feature_path = os.path.abspath("./vuvdetect_train_feature_tbd.mmap")
f0_path = os.path.abspath("./vuvdetect_train_f0_tbd.mmap")
n_sample = 65536 * 4
n_hop_per_sample = 256
min_break_hop = 32
fft_size = 256
n_bin = fft_size // 2 + 1
fft_freq_list = np.fft.rfftfreq(fft_size, d=1.0 / sr).astype(np.float32)
gvm_osr_list = [1, 1, 1, 1, 2, 4, 8, 16]
generate_uv = True

kernel_nyq = (1.0 / (hop_size / sr)) * 0.5
f0_kernel_min = 0.5 / kernel_nyq
f0_kernel_max = 5 / kernel_nyq
para_kernel_min = 1 / kernel_nyq
para_kernel_max = 5 / kernel_nyq
energy_kernel_min = 0.5 / kernel_nyq
energy_kernel_max = 1.5 / kernel_nyq

n_formant = 3

def generate_pseudo(lower, upper, n_hop, b, a):
  if lower == upper:
    return np.full(n_hop, lower, dtype=np.float32)
  l = np.random.uniform(-1.0, 1.0, size=n_hop)
  l = sp.filtfilt(b, a, l, method="gust").astype(np.float32)
  l -= np.min(l)
  m = np.max(l)
  if m == 0.0:
    l[:] = 1.0
  else:
    l /= np.max(l)
  l *= (upper - lower)
  l += lower
  return l

@nb.njit(fastmath=True)
def generate_worker_0(F_list, bw_list, amp_list, n_hop):
  freq_list = fft_freq_list[1:].copy()
  vt_magn_list = np.zeros((n_hop, n_bin), dtype=np.float32)
  for i_hop in range(n_hop):
    vt_magn_list[i_hop, 1:] = calcKlattFilterBankResponseMagnitude(freq_list, F_list[i_hop], bw_list[i_hop], amp_list[i_hop], sr)
  return vt_magn_list

@nb.njit(fastmath=True)
def generate_worker_1(rd_list, pulse_disto_list):
  (n_pulse,) = rd_list.shape
  (tp_list, te_list, ta_list) = np.zeros(n_pulse, dtype=np.float32), np.zeros(n_pulse, dtype=np.float32), np.zeros(n_pulse, dtype=np.float32)
  for i_pulse in range(n_pulse):
    (tp_list[i_pulse], te_list[i_pulse], ta_list[i_pulse]) = lfmodel.calcParameterFromRd(rd_list[i_pulse] * (1.0 - pulse_disto_list[i_pulse] * 0.5))
  return (tp_list, te_list, ta_list)

def generate_sample_0(freq_low, freq_high, rd_low, rd_high, disto_low, disto_high, energy_low, energy_high, magn_low, magn_high, snr_low, snr_high, use_shape_noise, osr, n_hop):
  f0_kernel_b, f0_kernel_a = sp.cheby1(4, 5, np.random.uniform(f0_kernel_min, f0_kernel_max))
  rd_kernel_b, rd_kernel_a = sp.cheby1(4, 5, np.random.uniform(para_kernel_min, para_kernel_max))
  disto_kernel_b, disto_kernel_a = sp.cheby1(4, 5, np.random.uniform(para_kernel_min, para_kernel_max))
  energy_kernel_b, energy_kernel_a = sp.cheby1(4, 5, np.random.uniform(energy_kernel_min, energy_kernel_max))
  snr_kernel_b, snr_kernel_a = sp.cheby1(4, 5, np.random.uniform(energy_kernel_min, energy_kernel_max))

  f0_list = generate_pseudo(freq_low, freq_high, n_hop, f0_kernel_b, f0_kernel_a)
  rd_list = generate_pseudo(rd_low, rd_high, n_hop, rd_kernel_b, rd_kernel_a)
  disto_list = generate_pseudo(disto_low, disto_high, n_hop, disto_kernel_b, disto_kernel_a)
  voiced_energy_list = generate_pseudo(energy_low, energy_high, n_hop, energy_kernel_b, energy_kernel_a)
  snr_list = generate_pseudo(snr_low, snr_high, n_hop, snr_kernel_b, snr_kernel_a)
  overall_magn = np.random.uniform(magn_low, magn_high)
  del f0_kernel_b, f0_kernel_a, rd_kernel_b, rd_kernel_a, disto_kernel_b, disto_kernel_a, energy_kernel_b, energy_kernel_a, snr_kernel_b, snr_kernel_a

  break_len = np.random.randint(30, 128)
  break_pos = np.random.randint(0, n_hop - break_len)
  f0_break_mode = np.random.randint(0, 4)
  if f0_break_mode == 0:
    f0_list[break_pos:break_pos + break_len] *= 0.5
  elif f0_break_mode == 1:
    f0_list[break_pos:break_pos + break_len] *= np.random.uniform(0.25, 2.0)
  f0_list[f0_list < band_freq_list[0]] = band_freq_list[0]
  f0_list[f0_list > band_freq_list[-1]] = band_freq_list[-1]
  del break_len, break_pos, f0_break_mode

  full_uv = generate_uv and (np.random.randint(0, 10) == 0)
  noise_energy_list = voiced_energy_list / snr_list
  if full_uv:
    noise_energy_list[:] = voiced_energy_list
    voiced_energy_list[:] = 0.0
    f0_list[:] = 0.0
  elif generate_uv:
    transProb = 0.01
    state = bool(np.random.randint(0, 2))
    keep = 12
    for i_hop in range(n_hop):
      if keep >= 12 and np.random.uniform(0.0, 1.0) < transProb:
        state = not state
        keep = 0
      if not state:
        f0_list[i_hop] = 0.0
        noise_energy_list[i_hop] = voiced_energy_list[i_hop]
      keep += 1
      del i_hop
    del keep, state

  F_list = np.zeros((n_hop, n_formant), dtype=np.float32)
  bw_list = np.zeros((n_hop, n_formant), dtype=np.float32)
  amp_list = np.zeros((n_hop, n_formant), dtype=np.float32)
  for i_formant in range(n_formant):
    b, a = sp.cheby1(4, 5, np.random.uniform(para_kernel_min, para_kernel_max))
    F_list[:, i_formant] = generate_pseudo(100.0, 1375.0, n_hop, b, a)
    b, a = sp.cheby1(4, 5, np.random.uniform(para_kernel_min, para_kernel_max))
    bw_list[:, i_formant] = generate_pseudo(100.0, 500.0, n_hop, b, a)
    b, a = sp.cheby1(4, 5, np.random.uniform(para_kernel_min, para_kernel_max))
    amp_list[:, i_formant] = np.exp(generate_pseudo(np.log(0.25), np.log(4), n_hop, b, a))
    del b, a
  vt_magn_list = generate_worker_0(F_list, bw_list, amp_list, n_hop)
  del F_list, bw_list, amp_list
  
  # Generate GVM Parameter
  n_sample = getNSample(n_hop, hop_size)
  (t_list, hop_idx_list) = gvm.convertF0ListToTList(f0_list, hop_size, sr)
  (n_pulse,) = hop_idx_list.shape
  T0_list = 1.0 / f0_list[hop_idx_list]
  pulse_disto_list = disto_list[hop_idx_list]
  pulse_energy_list = voiced_energy_list[hop_idx_list]
  (tp_list, te_list, ta_list) = generate_worker_1(rd_list[hop_idx_list], pulse_disto_list)
  if disto_low == disto_high == 0.0:
    pulse_vt_env_list = vt_magn_list[hop_idx_list]
  else:
    t_list, tp_list, te_list, ta_list, pulse_vt_env_list, pulse_energy_list = gvm.addDistortionEffect(pulse_disto_list, t_list, T0_list, tp_list, te_list, ta_list, vt_magn_list[hop_idx_list], pulse_energy_list, sr)
  
  if not full_uv:
    # Glottal Part
    if osr > 1:
      l = np.zeros((n_pulse, (pulse_vt_env_list.shape[1] - 1) * osr + 1), dtype=pulse_vt_env_list.dtype)
      for i, x in enumerate(pulse_vt_env_list):
        l[i, :pulse_vt_env_list.shape[1]] = x
        del i, x
      pulse_vt_env_list = l
      del l
    sinusoid, Ee_list = gvm.generateGlottalSource(n_sample * osr, t_list, T0_list, tp_list, te_list, ta_list, pulse_vt_env_list, pulse_energy_list, sr * osr)
    sinusoid = sp.resample_poly(sinusoid, 1, osr).astype(np.float32)
  else:
    Ee_list = np.zeros_like(t_list)
  
  # Noise Part
  if use_shape_noise:
    vuv_hop_list = f0_list > 0.0
    noise = gvm.generateNoisePart(n_sample, t_list, hop_idx_list, T0_list, Ee_list, tp_list, te_list, ta_list, vuv_hop_list, vt_magn_list, noise_energy_list, hop_size, sr)
    del vuv_hop_list
  else:
    noise = np.random.uniform(-1, 1, size=n_sample) * np.sqrt(3)
    noise_energy_list = ipl.interp1d(np.arange(n_hop) * hop_size, noise_energy_list, bounds_error=False, fill_value=noise_energy_list[-1], kind="linear")(np.arange(n_sample))
    noise *= noise_energy_list
    noise = noise.astype(np.float32)
  del noise_energy_list, Ee_list

  # Mix and normalize
  if not full_uv:
    out = sinusoid
    out += noise
    del sinusoid
  else:
    out = noise
  del noise

  out /= np.max(np.abs(out))
  out *= overall_magn

  # Fix f0_list from T0_list
  pulse_f0_list = 1.0 / T0_list
  x = t_list
  y = pulse_f0_list
  if not generate_uv:
    if x[0] != 0.0:
      x = np.concatenate(((0,), x))
      y = np.concatenate(((y[0],), y, ))
    if x[-1] != n_hop * hop_size / sr:
      x = np.concatenate((x, (n_hop * hop_size / sr,)))
      y = np.concatenate((y, (y[-1],)))
    f0_list = ipl.interp1d(x, y)(np.arange(n_hop) * hop_size / sr)

  return f0_list, out

@nb.njit(fastmath=True, cache=True)
def gen_if(x, out, i_freq, h1, hd1, h2, hd2, n_hop):
  (nh,) = h1.shape
  for i_hop in nb.prange(n_hop):
    i_center = int(round(i_hop * hop_size))
    frame = getFrame(x, i_center, nh)
    out[i_hop, i_freq, 3] = yang.calcYangIF(frame, h1, hd1)
    out[i_hop, i_freq, 4] = yang.calcYangIF(frame, h2, hd2)

def generate_sample_1(x: np.ndarray):
  (n_x,) = x.shape
  n_hop = getNFrame(n_x, hop_size)
  out = np.zeros((n_hop, n_band, 5), dtype=np.float32)
  idx_list = (np.arange(0, n_hop, dtype=np.float64) * hop_size).round().astype(np.int32)
  for (i_freq, freq) in enumerate(band_freq_list):
    rel_freq = freq / sr
    w = yang.createYangSNRParameter(rel_freq)
    out[:, i_freq, 0] = np.log(yang.calcYangSNR(x, rel_freq * 0.5, w, mt=False)[idx_list] + 1e-5)
    out[:, i_freq, 1] = np.log(yang.calcYangSNR(x, rel_freq, w, mt=False)[idx_list] + 1e-5)
    out[:, i_freq, 2] = np.log(yang.calcYangSNR(x, rel_freq * 2, w, mt=False)[idx_list] + 1e-5)
    del w

    h1, hd1 = yang.createYangIFParameter(rel_freq, rel_freq)
    h2, hd2 = yang.createYangIFParameter(rel_freq * 2, rel_freq)
    gen_if(x, out, i_freq, h1, hd1, h2, hd2, n_hop)
  return out

def generate_main(queue: mp.Queue, worker_id, start, step, n):
  print("[Worker %d] Spawn" % (worker_id,))
  wp.SetProcessAffinityMask(-1, 1<< worker_id)
  wp.SetPriorityClass(-1, 0x00000040 | 0x00100000)
  for i_sample in range(start, n, step):
    freq_low, freq_high = sorted(np.random.uniform(band_freq_list[0], band_freq_list[-1], size=2))
    rd_low, rd_high = sorted(np.random.uniform(0.3, 3.0, size=2))
    disto_low, disto_high = sorted(np.random.uniform(0.0, 0.1, size=2))
    energy_low, energy_high = sorted(np.random.uniform(0.025, 1.0, size=2))
    magn_low, magn_high = sorted(np.random.uniform(0.025, 1.0, size=2))
    snr_low, snr_high = np.exp(sorted(np.random.uniform(np.log(8), np.log(1e16), size=2)))

    '''
    rd_low, rd_high = 3.0, 3.0
    disto_low, disto_high = 0.0, 0.1
    energy_low, energy_high = 1.0, 1.0
    magn_low, magn_high = 1.0, 1.0
    snr_low, snr_high = 8, 8
    '''

    use_shape_noise = (np.random.randint(0, 2) == 1)
    n_extend_hop = np.random.randint(0, 32)

    #print("S0")
    osr = np.random.choice(gvm_osr_list)
    f0_list, w = generate_sample_0(freq_low, freq_high, rd_low, rd_high, disto_low, disto_high, energy_low, energy_high, magn_low, magn_high, snr_low, snr_high, use_shape_noise, osr, n_hop_per_sample + n_extend_hop)
    cut_begin = np.random.randint(0, n_extend_hop + 1)
    f0_list = f0_list[cut_begin:cut_begin + n_hop_per_sample]
    #print("S1")
    feature_list = generate_sample_1(w)[cut_begin:cut_begin + n_hop_per_sample]
    
    queue.put((i_sample, f0_list, feature_list))
    
    '''
    pl.figure()
    pl.imshow(feature_list[:, :, 1].T, interpolation='nearest', aspect='auto', origin='lower', cmap="plasma")
    pl.plot((freqToBark(f0_list) - band_start) / (band_stop - band_start) * n_band)
    pl.figure()
    pl.imshow(feature_list[:, :, 2].T, interpolation='nearest', aspect='auto', origin='lower', cmap="plasma")
    pl.plot((freqToBark(f0_list) - band_start) / (band_stop - band_start) * n_band)
    pl.figure()
    pl.imshow(feature_list[:, :, 3].T, interpolation='nearest', aspect='auto', origin='lower', cmap="plasma", vmin=band_freq_list[0] / sr, vmax=band_freq_list[-1] / sr)
    pl.plot((freqToBark(f0_list) - band_start) / (band_stop - band_start) * n_band)
    pl.show()
    '''
  
  queue.put(worker_id)

if __name__ == "__main__":
  print("* Mapping...")
  out_f0 = np.memmap(f0_path, dtype=np.float32, mode="w+", shape=(n_sample, n_hop_per_sample), order="C")
  out_feature = np.memmap(feature_path, dtype=np.float32, mode="w+", shape=(n_sample, n_hop_per_sample, n_band, 5), order="C")

  print("* Generate...")
  ctx = mp.get_context('spawn')
  queue = mp.Queue(16)

  n_chunk = 8
  chunk_size = n_sample // n_chunk
  chunk_list = []
  for i_chunk in range(n_chunk):
    p = ctx.Process(target=generate_main, args=(queue, i_chunk, i_chunk, n_chunk, n_sample))
    p.start()
    chunk_list.append(p)
  while any(x is not None for x in chunk_list):
    x = queue.get()
    if isinstance(x, int):
      print("Worker %d exited" % (x,))
      chunk_list[x] = None
    else:
      i_sample, f0_data, feature_data = x
      print("Write back %d/%d" % (i_sample, n_sample))
      out_f0[i_sample, :] = f0_data
      out_feature[i_sample, :, :, :] = feature_data
      del i_sample, f0_data, feature_data