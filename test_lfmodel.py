import numpy as np
import pylab as pl
from revoice import *
from revoice.common import *

sr = 48000
nSample = int(3.0 * sr)
T0 = 1.0 / 220.0
T1 = 1.0 / 440.0
Rd0 = 3 * (1 - 0.5)
Rd1 = 3 * (1 - 0.8)

assert (np.abs(np.array(lfmodel.calcLFModelParameterFromGFModel(T0, *lfmodel.calcGFModelParameterFromLFModel(T0, *lfmodel.calcParameterFromRd(0.3)))) - np.array(lfmodel.calcParameterFromRd(0.3))) < 1e-5).all()

print("Plotting...")
t = np.linspace(0, 0.008, 16384, dtype=np.float32)
f = np.linspace(1.0, 4000.0, 16384 * 16, dtype=np.float32)
f = np.fft.rfftfreq(16384, 0.008 / 16384)[1:].astype(np.float32)
T0 = 0.008

print(np.sum((lfmodel.calcFlowDerivative(t, T0, 1.0, *lfmodel.calcParameterFromRd(0.3)) / 16384 / 256 * 2) ** 2))
print(np.sum(np.abs(np.fft.rfft(lfmodel.calcFlowDerivative(t, T0, 1.0, *lfmodel.calcParameterFromRd(0.3)) / 16384 / 256 * 2)) ** 2) / 8192)
print(np.sum(np.abs(lfmodel.calcSpectrum(f, T0, 1.0, *lfmodel.calcParameterFromRd(0.3)))**2)/8192)

pl.figure()
pl.title("Glottal Derivative")
pl.plot(t, lfmodel.calcFlowDerivative(t, T0, 1.0, *lfmodel.calcParameterFromRd(0.3)), label = "Rd = 0.3")
pl.plot(t, lfmodel.calcFlowDerivative(t, T0, 1.0, *lfmodel.calcParameterFromRd(1.0)), label = "Rd = 1.0")
pl.plot(t, lfmodel.calcFlowDerivative(t, T0, 1.0, *lfmodel.calcParameterFromRd(2.5)), label = "Rd = 2.5")
pl.legend()

pl.figure()
pl.title("Glottal Openness")
pl.plot(t, lfmodel.calcGlottalOpenness(t, T0, 1.0, *lfmodel.calcParameterFromRd(0.3)), label = "Rd = 0.3")
pl.plot(t, lfmodel.calcGlottalOpenness(t, T0, 1.0, *lfmodel.calcParameterFromRd(1.0)), label = "Rd = 1.0")
pl.plot(t, lfmodel.calcGlottalOpenness(t, T0, 1.0, *lfmodel.calcParameterFromRd(2.5)), label = "Rd = 2.5")
pl.legend()

pl.figure()
pl.title("Magn")
pl.plot(f, np.log(np.abs(np.fft.rfft(lfmodel.calcFlowDerivative(t, T0, 1.0, *lfmodel.calcParameterFromRd(2.5)) / 16384 / 256 * 2)))[1:], label = "Rd = 2.5, FFT")
pl.plot(f, np.log(np.abs(lfmodel.calcSpectrum(f, T0, 1.0, *lfmodel.calcParameterFromRd(0.3)))), label = "Rd = 0.3")
pl.plot(f, np.log(np.abs(lfmodel.calcSpectrum(f, T0, 1.0, *lfmodel.calcParameterFromRd(1.0)))), label = "Rd = 1.0")
pl.plot(f, np.log(np.abs(lfmodel.calcSpectrum(f, T0, 1.0, *lfmodel.calcParameterFromRd(2.5)))), label = "Rd = 2.5")
pl.legend()
pl.show()