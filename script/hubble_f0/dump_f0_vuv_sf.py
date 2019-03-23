import sys
sys.path.append("../../")

import numpy as np
import pylab as pl
from revoice import *
from revoice.common import *
import revoice.tf_model

import os, time, queue, ctypes, pickle, threading
import tensorflow as tf
from config import *

with open("normalize.pickle", "rb") as f:
  (mean, stdev) = pickle.load(f)
mean = mean.reshape(n_band, 5)[:, (0, 1, 3)].copy()
stdev = stdev.reshape(n_band, 5)[:, (0, 1, 3)].copy()

path_checkpoint_dir = "tf_f0_vuv_ckpt_D128_D65"
tf_dtype = tf.float32

n_feature = n_band * 3

print("* Create session...")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
tf_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
sess = tf.Session(config=tf_config)

print("* Define network...")
p_input = tf.placeholder(dtype=tf_dtype, shape=[None, n_feature], name="data_input")
p_target = tf.placeholder(dtype=tf_dtype, shape=[None, n_band + 1], name="data_target")
v_net = revoice.tf_model.hubbleF0Network2(p_input, isTrain=False, reuse=False, scopeName="net_hubble_vuv")

print("* Initialize variable...")
op_init_var = tf.global_variables_initializer()
sess.run(op_init_var)
print("* Create saver...")
saver = tf.train.Saver()
print("* Load checkpoint...")
checkpoint = tf.train.get_checkpoint_state(path_checkpoint_dir)
if checkpoint and checkpoint.model_checkpoint_path:
  checkpoint_name = os.path.basename(checkpoint.model_checkpoint_path)
  saver.restore(sess, os.path.join(path_checkpoint_dir, checkpoint_name))
  print("=> Success")

  trainable_var_list = tf.trainable_variables()
  var_list = [var for var in trainable_var_list if "net_hubble_vuv" in var.name]
  var_dict = {}
  for var in trainable_var_list:
    var_dict[var.name] = sess.run(var)
else:
  print("=> Failed [!]")
  exit(1)

var_dict["mean"] = mean
var_dict["stdev"] = stdev
pickle.dump(var_dict, open("hubble_f0_vuv_sf.pickle", "wb"))