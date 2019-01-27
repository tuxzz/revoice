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

print("Generializing constant...")
saveWav("lf_variant.wav", lfmodel.calcVariantFlowDerivative(nSample, [0], [T0], [1.0], [Rd0], sr), sr)

print("Generializing variant...")
saveWav("lf_variant.wav", lfmodel.calcVariantFlowDerivative(nSample, [0, sr], [T0, T1], [1.0, 0.5], [Rd0, Rd1], sr), sr)

print("Plotting...")
t = np.linspace(0, 0.008, 16384)
f = np.linspace(1.0, 4000.0, 16384 * 16)
T0 = 0.008

pl.figure()
pl.subplot(111)
pl.plot(t, lfmodel.calcFlowDerivative(t, T0, 1.0, *lfmodel.calcParameterFromRd(0.3)), label = "Rd = 0.3")
pl.plot(t, lfmodel.calcFlowDerivative(t, T0, 1.0, *lfmodel.calcParameterFromRd(1.0)), label = "Rd = 1.0")
pl.plot(t, lfmodel.calcFlowDerivative(t, T0, 1.0, *lfmodel.calcParameterFromRd(2.5)), label = "Rd = 2.5")
pl.legend()
pl.figure()
pl.plot(f, np.log(np.abs(lfmodel.calcSpectrum(f, T0, 1.0, *lfmodel.calcParameterFromRd(0.3)))), label = "Rd = 0.3")
pl.plot(f, np.log(np.abs(lfmodel.calcSpectrum(f, T0, 1.0, *lfmodel.calcParameterFromRd(1.0)))), label = "Rd = 1.0")
pl.plot(f, np.log(np.abs(lfmodel.calcSpectrum(f, T0, 1.0, *lfmodel.calcParameterFromRd(2.5)))), label = "Rd = 2.5")
pl.legend()
pl.show()