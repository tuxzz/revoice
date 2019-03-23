# REVOICE
**Just a voice analysis/synthesis algorithm pack.
On the way of approaching it.**

## Developer
Tuku <tuku@tuxzz.org>  
Hikaria <hikaria@tuxzz.org>

## Done
* [x] Ugly and simple viterbi decoder(sparsehmm)
* [x] LPC(lpc)
* [x] Yin F0 estimator(yin)
* [x] PYin F0 estimator(pyin)
* [x] Viterbi F0 tracker(monopitch)
* [x] Viterbi note tracker(mononote)
* [x] Adaptive STFT(adaptivestft)
* [x] Cheaptrick Envelope(cheaptrick)
* [x] MFI Envelope(mfienvelope)
* [x] HNM analyzer framework and synther(hnm)
* [x] HNM analyzer based on STFT(hnm_qfft)
* [x] HNM analyzer based on QHM-AIR(hnm_qhm)
* [x] HNM analyzer based on signle band DFT(hnm_get)
* [x] LF-Model(lfmodel)
* [x] Magnitude-only Rd estimator(rd_krh)
* [x] Voice Tract Model(vt)
* [x] (Decaparted)Ugly and simple formant tracker(formanttracker)
* [x] Yang style SNR and instantaneous frequency(yang)
* [x] Partial Glottal-Tract model(gvm)
* [x] Hubble F0 detect system(hubble)

## Future scope
* [ ] Use log probabilities in sparsehmm
* [ ] SRH F0 estimator
* [ ] STFChT
* [ ] STFChT F0 estimator
* [ ] HNM based on STFChT
* [ ] DCE-MFA Envelope
* [ ] Formant Refinement
* [ ] Voice transformation
* [ ] Harmonic to noise conversion
* [ ] Better Documention

## Useful functions in `common.py`
* [x] Create an aliasing-free, non-integer offset and width hanning window via frequency domain method(accurateHann, fdHann)
* [x] Denoise data and keep instantaneous information(applySmoothingFilter)
* [x] Do DFT at any frequency for input signal(calcSpectrumAtFreq)
* [x] Measure difference between two spectrum magnitude(calcItakuraSaitoDistance)
* [x] Measure similarity between two time domain signal(calcSRER)

## Works cited
1. Mauch, Matthias, and Simon Dixon. "pYIN: A fundamental frequency estimator using probabilistic threshold distributions." 2014 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2014.
2. Morise, Masanori. "CheapTrick, a spectral envelope estimator for high-quality speech synthesis." Speech Communication 67 (2015): 1-7.
3. Nakano, Tomoyasu, and Masataka Goto. "A spectral envelope estimation method based on F0-adaptive multi-frame integration analysis." SAPA@ INTERSPEECH. 2012.
4. Fant, Gunnar. "The LF-model revisited. Transformations and frequency domain analysis." Speech Trans. Lab. Q. Rep., Royal Inst. of Tech. Stockholm 2.3 (1995): 40.
5. Bonada, Jordi. "High quality voice transformations based on modeling radiated voice pulses in frequency domain." Proc. Digital Audio Effects (DAFx). Vol. 3. 2004.
6. Saratxaga, Ibon, et al. "Simple representation of signal phase for harmonic speech models." Electronics letters 45.7 (2009): 381-383.
7. Story, Brad H. "A parametric model of the vocal tract area function for vowel and consonant simulation." The Journal of the Acoustical Society of America 117.5 (2005): 3231-3254.
8. Kawahara, H., Y. Agiomyrgiannakis, and H. Zen. "YANG vocoder, Google."
9. Hua, Kanru. "Nebula: F0 Estimation and Voicing Detection by Modeling the Statistical Properties of Feature Extractors." arXiv preprint arXiv:1710.11317 (2017).