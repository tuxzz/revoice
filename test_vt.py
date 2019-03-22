import numpy as np
import pylab as pl
from revoice import *
from revoice.common import *

sr = 48000
nSample = int(3.0 * sr)
T = 1.0 / 220.0
Rd = lfmodel.calcRdFromTenseness(0.625)

print("Generializing Glottal Source...")
glottal_src = lfmodel.calcVariantFlowDerivative(nSample, [0, sr], [T, T * 0.5], [0.9, 0.5], [Rd, Rd * 1.25], sr)
assert glottal_src.dtype == np.float32
saveWav("vt_src.wav", glottal_src, sr)

print("Apply Voice Tract...")
vtArea = np.array([0.36,0.36,0.36,0.36,0.36,0.36,0.36,1.21,1.21,1.21,2.2944685016355075,2.2031707757513903,2.1121126512026893,2.0262478049572152,1.94742105117564,1.8772024817010853,1.8168851099407863,1.7674957113823353,1.7298144161169589,1.7043981782451614,1.6916033665720265,1.6916033665720265,1.7043981782451614,1.7298144161169589,1.7674957113823353,1.8168851099407863,1.8772024817010853,1.94742105117564,2.0262478049572152,2.1121126512026893,2.2031707757513903,2.2973216906248046,2.3922470878297757,2.4854679237953228,2.5744193288347783,2.6565401396973054,2.729372268660823,2.756583972807105,2.715313142404605,2.25,2.25,2.25,2.25,2.25,], dtype=np.float32)
noseArea = np.array([0.16,0.2644897959183674,0.3951020408163265,0.5518367346938776,0.7346938775510206,0.9436734693877553,1.178775510204082,1.44,1.7273469387755103,2.040816326530613,2.380408163265307,2.7461224489795915,3.1379591836734693,3.555918367346939,3.61,3.582908163265306,3.188775510204082,2.8176020408163276,2.4693877551020402,2.144132653061224,1.8418367346938778,1.5625,1.3061224489795917,1.0727040816326532,0.8622448979591839,0.6747448979591835,0.510204081632653,0.3686224489795918,], dtype=np.float32)
print("* Voice Tract Length = %fm" % (vt.calcVoiceTractLength(vtArea.shape[0], sr * 2),))
out = sp.resample_poly(vt.applyVoiceTractFilterTwoTract(sp.resample_poly(glottal_src, 2, 1), vtArea, noseArea, 0.16, 0.75, -0.85), 1, 2).astype(np.float32)
saveWav("vt_target.wav", out, sr)

pl.figure()
pl.plot(vtArea)

print("Get impulse response...")
pl.figure()

n = 512
fdGlottalSrc = np.abs(lfmodel.calcSpectrum(np.arange(1, n // 2 + 1, dtype=np.float32) / n * sr, T, 1.0, *lfmodel.calcParameterFromRd(Rd)), dtype=np.float32)
fdGlottalSrc = np.concatenate((np.array((1e-5,), dtype=np.float32), fdGlottalSrc))
assert fdGlottalSrc.dtype == np.float32
fdImpulse = np.abs(np.fft.rfft(vt.applyVoiceTractFilterSingleTractALP(sp.unit_impulse(n), vtArea, 0.75, -0.85))) * fdGlottalSrc
pl.plot(np.log(fdImpulse))

fdImpulse = np.abs(np.fft.rfft(sp.resample_poly(vt.applyVoiceTractFilterSingleTract(sp.unit_impulse(n * 2), vtArea, 0.75, -0.85), 1, 2))) * fdGlottalSrc
pl.plot(np.log(fdImpulse))

fdImpulse = np.abs(np.fft.rfft(sp.resample_poly(vt.applyVoiceTractFilterTwoTract(sp.unit_impulse(n * 2), vtArea, noseArea, 0.5, 0.75, -0.85), 1, 2))) * fdGlottalSrc
pl.plot(np.log(fdImpulse))

pl.show()