import multiprocessing as mp
if __name__ == "__main__":
  mp.set_start_method("forkserver")

import sys
sys.path.append("../../")

import numpy as np
import pylab as pl
from revoice import *
from revoice.common import *
import revoice.tf_model

import os, time, queue, ctypes, pickle, threading
import multiprocessing as mp
import tensorflow as tf

import ctypes

from config import *

with open("normalize.pickle", "rb") as f:
  (mean, stdev) = pickle.load(f)
mean = mean.reshape(n_band, 5)[:, (0, 1, 3)].copy()
stdev = stdev.reshape(n_band, 5)[:, (0, 1, 3)].copy()

tf_dtype = tf.float32
np_dtype = np.float32
learning_rate = 1e-4
n_epoch = 2048
batch_size = 16384
checkpoint_interval = 256
path_checkpoint_dir = "tf_f0_vuv_ckpt_D128_D65"
n_data_process = 4
test_mode = True

f0_path = os.path.abspath("/home/tuxzz/vuvdetect_train_f0_marcob.mmap")
feature_path = os.path.abspath("/home/tuxzz/vuvdetect_train_feature_marcob_nh3.mmap")

n_feature = n_band * 3

print("* Mapping...")
f0_data = np.memmap(f0_path, dtype=np.float32, mode="r", order="C")
feature_data = np.memmap(feature_path, dtype=np.float32, mode="r", order="C")

if sys.platform == "linux" or sys.platform == "linux2":
  madvise = ctypes.CDLL("libc.so.6").madvise
  madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
  madvise.restype = ctypes.c_int
  print("MADVISE:", madvise(f0_data.ctypes.data, f0_data.size * f0_data.dtype.itemsize, 1))
  print("MADVISE:", madvise(feature_data.ctypes.data, feature_data.size * feature_data.dtype.itemsize, 1))

n_sample = f0_data.shape[0]
f0_data = f0_data.reshape(n_sample)
feature_data = feature_data.reshape(n_sample, n_feature)
print("Data shape = %s" % (str(f0_data.shape),))

@nb.njit()
def normalize_data_inplace(feature_list):
  feature_list -= mean.reshape(1, n_feature)
  feature_list /= stdev.reshape(1, n_feature)

@nb.njit()
def fetch_data():
  sample_idx_list = np.random.randint(0, n_sample, size=batch_size)
  
  f0_list = f0_data[sample_idx_list]
  f0_map_list = np.zeros((batch_size, n_band + 1), dtype=np.float32)
  for i_batch in range(batch_size):
    if f0_list[i_batch] <= 0.0:
      f0_map_list[i_batch, -1] = 1.0
    else:
      f0_map_list[i_batch, np.argmin(np.abs(f0_list[i_batch] - band_freq_list))] = 1.0
  feature_list = feature_data[sample_idx_list].copy()
  normalize_data_inplace(feature_list)
  return f0_map_list, feature_list

def data_fetch_process_main(target_queue, memory_queue, command_queue, memory_pool):
  while True:
    i_memory = memory_queue.get()
    (memory_f0, memory_feature) = memory_pool[i_memory]
    with memory_f0.get_lock():
      with memory_feature.get_lock():
        np.frombuffer(memory_f0.get_obj(), dtype=np.bool).reshape(batch_size, n_band + 1)[:], np.frombuffer(memory_feature.get_obj(), dtype=np.float32).reshape(batch_size, n_feature)[:] = fetch_data()
    target_queue.put(i_memory)
    try:
      command_queue.get_nowait()
      return
    except queue.Empty:
      pass

if __name__ == "__main__":
  print("* Create session...")
  if test_mode:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
  tf_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
  tf_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
  sess = tf.Session(config=tf_config)

  print("* Define network...")
  if test_mode:
    p_input = tf.placeholder(dtype=tf_dtype, shape=[None, n_feature], name="data_input")
    p_target = tf.placeholder(dtype=tf_dtype, shape=[None, n_band + 1], name="data_target")
  else:
    p_input = tf.placeholder(dtype=tf_dtype, shape=[batch_size, n_feature], name="data_input")
    p_target = tf.placeholder(dtype=tf_dtype, shape=[batch_size, n_band + 1], name="data_target")
  v_logits, v_net = revoice.tf_model.hubbleF0Network2(p_input, isTrain=not test_mode, reuse=False, scopeName="net_hubble_vuv")
  v_loss = revoice.tf_model.f0CostFunction(p_target, v_logits)

  print("* Create saver...")
  saver = tf.train.Saver()
  print("* Create optimizer...")
  trainable_var_list = tf.trainable_variables()
  var_list = [var for var in trainable_var_list if "net_hubble_vuv" in var.name]
  print("=> Trainable variable list:")
  for var in trainable_var_list:
    print("==>", var.name)
  op_optimizer = tf.contrib.opt.AdamWOptimizer(learning_rate, epsilon=1e-4).minimize(v_loss, var_list=var_list)
  print("* Initialize variable...")
  op_init_var = tf.global_variables_initializer()
  sess.run(op_init_var)

  print("* Load checkpoint...")
  checkpoint = tf.train.get_checkpoint_state(path_checkpoint_dir)
  if checkpoint and checkpoint.model_checkpoint_path:
    checkpoint_name = os.path.basename(checkpoint.model_checkpoint_path)
    saver.restore(sess, os.path.join(path_checkpoint_dir, checkpoint_name))
    print("=> Success")
  else:
    print("=> Failed [!]")

  if test_mode:
    print("* Load test data...")
    @nb.njit(fastmath=True, cache=True)
    def gen_if(x, out, i_freq, h1, hd1, h2, hd2, n_hop):
      (nh,) = h1.shape
      for i_hop in nb.prange(n_hop):
        i_center = int(round(i_hop * hop_size))
        frame = getFrame(x, i_center, nh)
        out[i_hop, i_freq, 2] = yang.calcYangIF(frame, h1, hd1)
        #out[i_hop, i_freq, 4] = yang.calcYangIF(frame, h2, hd2)
    def generate_sample_1(x: np.ndarray, mt: bool = False):
      (n_x,) = x.shape
      n_hop = getNFrame(n_x, hop_size)
      out = np.zeros((n_hop, n_band, 3), dtype=np.float32)
      idx_list = (np.arange(0, n_hop, dtype=np.float64) * hop_size).round().astype(np.int32)
      for (i_freq, freq) in enumerate(band_freq_list):
        rel_freq = freq / sr
        w = yang.createYangSNRParameter(rel_freq)
        out[:, i_freq, 0] = np.log(yang.calcYangSNR(x, rel_freq * 0.5, w, mt=mt)[idx_list] + 1e-5)
        out[:, i_freq, 1] = np.log(yang.calcYangSNR(x, rel_freq, w, mt=mt)[idx_list] + 1e-5)
        #out[:, i_freq, 2] = np.log(yang.calcYangSNR(x, rel_freq * 2, w, mt=mt)[idx_list] + 1e-5)
        del w

        h1, hd1 = yang.createYangIFParameter(rel_freq, rel_freq)
        h2, hd2 = yang.createYangIFParameter(rel_freq * 2, rel_freq)
        gen_if(x, out, i_freq, h1, hd1, h2, hd2, n_hop)
      return out
    def worker():
      x, work_sr = loadWav("../../voices/yuri_orig.wav")
      if x.ndim == 2:
        x = x.T[0]
      if work_sr != sr:
        x = sp.resample_poly(x, sr, work_sr).astype(np.float32)
      b, a = dc_iir_filter(50.0 / sr / 2)
      x = sp.filtfilt(b, a, x).astype(np.float32)
      return generate_sample_1(x, mt=True), energy.Analyzer(sr)(x)
    (feature_list, energy_list) = worker()
    (n_hop, _, _) = feature_list.shape
    feature_list = feature_list.reshape(n_hop, n_feature)
    normalize_data_inplace(feature_list)
    print("* Do test...")
    f0_map_list = sess.run(v_net, feed_dict={p_input: feature_list}).reshape(n_hop, n_band + 1)
    tracker = revoice.tf_model.HubbleTracker()
    path = tracker(f0_map_list)
    path = path.astype(np.float64)
    path[path >= n_band] = np.nan
    result_path = path.copy()
    result_path[energy_list < 1e-8] = np.nan
    result_vuv = ~np.isnan(result_path)
    with open("result_label.txt", "wb") as f:
      if result_vuv[0]:
        f.write("0.0\t0.0\tV\n".encode("UTF-8"))
      else:
        f.write("0.0\t0.0\tU\n".encode("UTF-8"))
      for i in range(1, result_path.size):
        if result_vuv[i - 1] != result_vuv[i]:
          t = i * hop_size / sr
          if result_vuv[i]:
            f.write(("%f\t%f\tV\n" % (t, t)).encode("UTF-8"))
          else:
            f.write(("%f\t%f\tU\n" % (t, t)).encode("UTF-8"))
    pl.figure()
    pl.title("SNR0")
    pl.imshow(feature_list.reshape(n_hop, n_band, 3)[:, :, 0].T, cmap='plasma', interpolation='nearest', aspect='auto', origin='lower')
    pl.plot(np.argmax(f0_map_list, axis=1), 'g', label="Simple")
    pl.plot(path, 'cx', label="Viterbi")
    pl.plot(result_path, 'rx', label="Result")
    pl.figure()
    pl.title("SNR1")
    pl.imshow(feature_list.reshape(n_hop, n_band, 3)[:, :, 1].T, cmap='plasma', interpolation='nearest', aspect='auto', origin='lower')
    pl.plot(np.argmax(f0_map_list, axis=1), 'g', label="Simple")
    pl.plot(path, 'cx', label="Viterbi")
    pl.plot(result_path, 'rx', label="Result")
    pl.figure()
    pl.title("Posterior Probability Map")
    pl.imshow(np.log(np.clip(f0_map_list.reshape(n_hop, n_band + 1).T, 1e-5, 1.0)), cmap='plasma', interpolation='nearest', aspect='auto', origin='lower')
    pl.plot(np.argmax(f0_map_list, axis=1), 'g', label="Simple")
    pl.plot(path, 'cx', label="Viterbi")
    pl.plot(result_path, 'rx', label="Result")
    pl.show()

  else:
    print("* Spawn data process...")
    ctx = mp.get_context('spawn')
    memory_pool = []
    data_queue = mp.Queue(n_data_process * 2)
    memory_queue = mp.Queue(n_data_process * 2)
    command_queue = mp.Queue(n_data_process)
    for i in range(n_data_process * 2):
      memory_pool.append((mp.Array(ctypes.c_bool, batch_size * (n_band + 1)), mp.Array(ctypes.c_float, batch_size * n_feature)))
      memory_queue.put(i)
      del i
    for _ in range(n_data_process):
      ctx.Process(target=data_fetch_process_main, args=(data_queue, memory_queue, command_queue, memory_pool)).start()

    def fetch_mem_data():
      i_memory = data_queue.get()
      (memory_f0, memory_feature) = memory_pool[i_memory]
      with memory_f0.get_lock():
        with memory_feature.get_lock():
          out = np.frombuffer(memory_f0.get_obj(), dtype=np.bool).reshape(batch_size, n_band + 1).copy(), np.frombuffer(memory_feature.get_obj(), dtype=np.float32).reshape(batch_size, n_feature).copy()
      memory_queue.put(i_memory)
      return out

    print("Train...")
    training_start_time = time.time()
    for i_epoch in range(n_epoch):
      epoch_start_time = time.time()
      target_list, input_list = fetch_mem_data()
      #target_list, input_list = fetch_data()
      #print(time.time() - epoch_start_time)
      
      '''
      pl.figure()
      pl.imshow(input_list.reshape(batch_size, n_band, 3)[:, :, 1].T, cmap='plasma', interpolation='nearest', aspect='auto', origin='lower')
      pl.plot(np.argmax(target_list, axis=1), "cx")
      pl.figure()
      pl.imshow(target_list.reshape(batch_size, n_band + 1).T, cmap='plasma', interpolation='nearest', aspect='auto', origin='lower')
      pl.plot(np.argmax(target_list, axis=1), "cx")
      pl.show()
      '''
      
      _, loss = sess.run([op_optimizer, v_loss], feed_dict = {p_input: input_list, p_target: target_list})
      t = time.time()
      t_epoch = t - epoch_start_time
      t_total = t - training_start_time
      print("Epoch %d: loss = %.4lf, tEpoch = %.6lf, tTotal = %.6lf" % (i_epoch + 1, loss, t_epoch, t_total))

      if i_epoch != 0 and (i_epoch % checkpoint_interval == 0 or i_epoch == n_epoch - 1):
        print("* Save checkpoint...")
        if not os.path.exists(path_checkpoint_dir):
          os.makedirs(path_checkpoint_dir)
        saver.save(sess, os.path.join(path_checkpoint_dir, "hubble_vuv"), global_step=i_epoch + 1)

    for _ in range(n_data_process):
      command_queue.put(None)
    while not data_queue.empty():
      data_queue.get()