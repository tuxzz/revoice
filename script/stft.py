import sys
sys.path.append("../")

from revoice import *
from revoice.common import *

import numpy as np
import pylab as pl

w, sr = loadWav("../voices/yuri_orig.wav")

b, a = dc_iir_filter(50.0 / sr / 2.0)
w = sp.filtfilt(b, a, w).astype(np.float32)

window = nuttall(2048, nuttall_min3_coeff, sym=False)
hop_size = list_stft_hop_size(window)[-1]
assert sp.check_COLA(window, window.size, window.size - hop_size)
assert sp.check_NOLA(window, window.size, window.size - hop_size)
spec = stft(w, window, hop_size, 4096, remove_dc=True)
reconstruct_w = istft(spec, istft_window(window, hop_size), hop_size, w.shape[0])

pl.figure()
pl.plot(w)
pl.plot(reconstruct_w)
pl.figure()
pl.imshow(np.log(np.abs(spec) + 1e-5).T, interpolation='nearest', aspect='auto', origin='lower', vmin=(-80.0 / 20.0) / np.log10(np.e), vmax=(0.0 / 20.0) / np.log10(np.e))
pl.show()