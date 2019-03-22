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

## On the way
* [ ] Hubble F0 detect system

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