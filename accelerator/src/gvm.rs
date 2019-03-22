use num_complex::*;

struct LFInternalParameter {
  wg: f32,
  sin_wg_te: f32,
  cos_wg_te: f32,
  e: f32,
  a_u: f32,
  a: f32,
  e0: f32,
}

fn calc_internal_parameter(t0: f32, ee: f32, tp: f32, te: f32, ta: f32) -> LFInternalParameter {
  assert!(t0 > 0.0 && ee >= 0.0 && tp > 0.0 && te > 0.0 && ta > 0.0, "invalid input");
  let wg = std::f32::consts::PI / tp;
  let sin_wg_te = (wg * te).sin();
  let cos_wg_te = (wg * te).cos();
  let mut e = ((-ta * fastapprox::fast::lambertw((te - t0) * ((te - t0) / ta).exp() / ta) + te - t0) / (ta * te - ta * t0)).max(1.0);
  if e.is_nan() || e.is_infinite() || e > 2.0 / (ta + 1e-9) {
    println!("[WARNING] Failed to solve LF parameter `e`, got e = {}", e);
    e = 1.0;
  }

  let e_te_t0 = (e * (te - t0)).exp();
  let a_u = (1.0 - e_te_t0) / (e * e * ta) + (te - t0) * e_te_t0 / (e * ta);
  let mut a = wg * (a_u * sin_wg_te * wg - cos_wg_te + 1.0);
  for _ in 0..8 {
    a = a - ((a * a + wg * wg) * sin_wg_te * a_u + wg * ((-a * te).exp() - cos_wg_te) + a * sin_wg_te) / (sin_wg_te * (2.0 * a_u * a + 1.0) - wg * te * (-te * a).exp());
  }
  if a.is_nan() || a.is_infinite() || a > 1e9 {
    println!("[WARNING] Failed to solve LF parameter `a`, got a = {}", a);
    a = 0.0;
  }
  let e0 = -ee / ((a * te).exp() * sin_wg_te);

  LFInternalParameter { wg, sin_wg_te, cos_wg_te, e, a_u, a, e0 }
}

fn calc_glottal_spectrum(f_list: &[f32], t0: f32, ee: f32, mut tp: f32, mut te: f32, mut ta: f32, out: &mut [Complex32]) {
  assert_eq!(f_list.len(), out.len());
  tp *= t0;
  te *= t0;
  ta *= t0;
  let LFInternalParameter { wg, sin_wg_te, cos_wg_te, e, a_u: _, a, e0 } = calc_internal_parameter(t0, ee, tp, te, ta);
  out.iter_mut().enumerate().for_each(|(i, v)| {
    let f = f_list[i];
    let w = 2.0 * std::f32::consts::PI * f;
    let r = Complex::new(a, -w);
    let p1 = e0 / (r * r + wg * wg);
    let p2 = wg + (r * te).exp() * (r * sin_wg_te - wg * cos_wg_te);
    let p3 = ee * Complex::new(0.0, -w * te).exp() / (Complex::new(0.0, w * e * ta) * Complex::new(e, w));
    let p4 = e * (1.0 - e * ta) * (1.0 - Complex::new(0.0, -w * (t0 - te)).exp()) - Complex::new(0.0, w * e * ta);
    *v = p1 * p2 + p3 * p4;
  });
}

fn calc_glottal_openness(t_list: &[f32], t0: f32, ee: f32, mut tp: f32, mut te: f32, mut ta: f32, out: &mut [f32]) {
  assert_eq!(t_list.len(), out.len());
  tp *= t0;
  te *= t0;
  ta *= t0;
  let LFInternalParameter { wg, sin_wg_te, cos_wg_te, e, a_u: _, a, e0 } = calc_internal_parameter(t0, ee, tp, te, ta);
  let tr = te - t0;
  out.iter_mut().enumerate().for_each(|(i, v)| {
    let t = t_list[i];
    *v = if t <= te {
      e0 * ((a * t).exp() * (a * (wg * t).sin() - wg * (wg * t).cos()) + wg) / (a * a + wg * wg)
    }
    else if t <= t0 && t > te {
      let u = te - t;
      let c = ee * (e * u * (e * tr).exp() - (e * u).exp() + 1.0) / (ta * e * e);
      let vte = e0 * ((a * te).exp() * (a * (wg * te).sin() - wg * (wg * te).cos()) + wg) / (a * a + wg * wg);
      vte - c
    }
    else { 0.0 }
  });
}

fn generate_glottal_source(nSample, tList, T0List, tpList, teList, taList, pulseFilterList, energyList, sr):
  assert tList.shape == T0List.shape == tpList.shape == teList.shape == taList.shape == (pulseFilterList.shape[0],) == energyList.shape
  assert pulseFilterList.shape[1] % 2 == 1

  (nPulse, nBin) = pulseFilterList.shape
  fftSize = (nBin - 1) * 2
  f = (np.arange(0, nBin, dtype=np.float32) / fftSize * sr).astype(np.float32)
  f[0] = 1.0

  out = np.zeros(nSample, dtype=np.float32)
  EeList = np.zeros(nPulse, dtype=np.float32)
  for (iPulse, t) in enumerate(tList):
    T0 = T0List[iPulse]
    if T0 * sr > (fftSize // 2 - 1):
      raise ValueError("Frequency %f is too low, minimum acceptable is %f" % (1 / T0, sr / (fftSize // 2 - 1)))
    energy = energyList[iPulse]
    tp, te, ta = tpList[iPulse], teList[iPulse], taList[iPulse]
    pulseFilter = pulseFilterList[iPulse]
    fac = np.sqrt(np.mean(pulseFilter[1:] ** 2))
    if fac == 0.0:
      continue
    pulseFilter = pulseFilter / fac

    windowSize = int(2 * T0 * sr)
    if windowSize % 2 == 1:
      windowSize += 1
    window = np.hanning(windowSize)
    #windowNormFac = 1 / np.mean(window)

    iCenter = int((t + 0.5 * T0) * sr)
    tErr = (t + 0.5 * T0) - iCenter / sr
    #print(tErr)
    fdPulse = lfmodel.calcSpectrum(f, T0, 1.0, tp, te, ta)
    fdPulse[0] = 0.0
    fdPulse *= np.exp(-1j * tErr * 2 * np.pi * f)
    fdPulse += fdPulse * np.exp(-1j * T0 * 2 * np.pi * f)
    fdPulse *= pulseFilter

    #pl.plot(np.fft.irfft(fdPulse))
    #pl.show()
    
    tdPulse = np.fft.irfft(fdPulse)[:windowSize]
    '''td = np.zeros(windowSize, dtype=np.float32)
    td[:windowSize // 4] = tdPulse[windowSize - windowSize // 4:]
    td[windowSize // 4:windowSize // 4 + windowSize // 2] = tdPulse[:windowSize // 2]
    td[windowSize // 4 + windowSize // 2:] = tdPulse[windowSize // 2:windowSize - windowSize // 4]'''
    td = tdPulse
    tdPulse = td * window * sr
    #pl.plot(tdPulse)
    #pl.show()
    synthedEnergy = np.mean(tdPulse ** 2) * 4
    
    Ee = np.sqrt(energy / synthedEnergy)
    #print(np.sum(pulseFilter ** 2))
    #print(Ee)

    ob, oe, ib, ie = getFrameRange(nSample, iCenter, windowSize)
    out[ib:ie] += tdPulse[ob:oe] * Ee
    EeList[iPulse] = Ee

  return out, EeList