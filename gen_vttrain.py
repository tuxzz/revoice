import numpy as np
import pylab as pl
from revoice import *
from revoice.common import *
import time
from tensorlayer.prepro import threading_data
import os, pickle

def analyzeFor(path):
    s = time.time()
    def printp(s, *args, **kwargs):
        print("%s: %s" % (path, s), *args, **kwargs)
    
    printp("Load file...")
    w, sr = loadWav(path)

    printp("Resampling...")
    w = sp.resample_poly(w, int(np.round(w.shape[0] / sr * 48000)), w.shape[0])
    sr = 48000

    printp("Energy Analyzing...")
    energyList = energy.Analyzer(sr)(w)

    printp("F0 Estimation...")
    pyinAnalyzer = pyin.Analyzer(sr)
    obsProbList = pyinAnalyzer(w)
    monopitchAnalyzer = monopitch.Analyzer(*monopitch.parameterFromPYin(pyinAnalyzer))
    f0List = monopitchAnalyzer(obsProbList)
    silentList = energyList < 1e-8
    f0List[silentList] = -np.abs(f0List[silentList])

    printp("F0 Refinement...")
    f0RefineProcessor = refinef0.Processor(sr)
    f0List = f0RefineProcessor(w, f0List)

    printp("HNM Analyzing...")
    hnmAnalyzer = hnm.Analyzer(sr, harmonicAnalysisMethod = "qfft")
    hFreqList, hAmpList, hPhaseList, sinusoidEnergyList, noiseEnvList, noiseEnergyList = hnmAnalyzer(w, f0List)
    f0List = hFreqList[:, 0]

    printp("HNM Synthing...")
    synProc = hnm.Synther(sr)
    sinusoid = synProc(hFreqList, hAmpList, hPhaseList, sinusoidEnergyList, noiseEnvList, noiseEnergyList, enableNoise = False)

    printp("STFT Analyzing...")
    stftAnalyzer = adaptivestft.Analyzer(sr, fftSize = 1536)
    fSigList = stftAnalyzer(sinusoid, f0List)
    magnList = np.abs(fSigList)

    printp("Envelope Analyzing...")
    envAnalyzer = cheaptrick.Analyzer(sr)
    envList = envAnalyzer(magnList, f0List)
    envList = envList[:, 1:513]

    printp("Rd Analyzing...")
    nHop = f0List.shape[0]
    rdList = np.zeros(nHop)
    gainList = np.zeros(nHop)
    rdAnalyzer = rd_krh.Analyzer()
    for iHop, f0 in enumerate(f0List):
        if(f0 <= 0):
            continue
        need = hFreqList[iHop] > 0
        rdList[iHop], gainList[iHop] = rdAnalyzer(hFreqList[iHop][need], hAmpList[iHop][need])
    
    printp("LF Spectrum Analyzing...")
    lfMagnList = np.zeros(envList.shape)
    fList = rfftFreq(1536, sr)[1:513]
    for iHop, f0 in enumerate(f0List):
        if(f0 <= 0):
            continue
        lfMagnList[iHop] = np.abs(lfmodel.calcSpectrum(fList, 1 / f0, 1.0, *lfmodel.calcParameterFromRd(rdList[iHop]))) * gainList[iHop]
    
    printp("Normalizing...")
    need = f0List > 0.0
    envList[need] = np.log(np.exp(envList[need]) / lfMagnList[need])
    envList[need] /= np.sqrt(np.mean(envList[need] ** 2))
    
    printp("Slicing...")
    nHop = f0List.shape[0]
    sliceList = []
    for iHop, f0 in enumerate(f0List):
        if(f0 <= 0 or iHop < 1 or iHop > nHop - 2):
            continue
        if(f0List[iHop - 1] <= 0 or f0List[iHop + 1] <= 0):
            continue
        sliceList.append(np.concatenate((envList[iHop - 1], envList[iHop], envList[iHop + 1])).reshape(3, 512))
    sliceList = np.array(sliceList, dtype = np.float32)
    printp("Got %d slices, Time usage = %lfs." % (sliceList.shape[0], time.time() - s))

    return sliceList
   
analyzeFor("./vttrain/B\Gakkou_Kurumi.412.wav")
'''
srcDir = "./vttrain/B"

o = threading_data([os.path.join(srcDir, x) for x in os.listdir(srcDir)], analyzeFor)
o = np.concatenate(o)
with open("vttrain_B.pickle", "wb") as f:
    pickle.dump(o, f)
'''