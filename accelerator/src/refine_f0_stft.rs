use num_complex::*;
use fftw::plan::{R2CPlan, R2CPlan32};
use fftw::types::Flag;
use rayon::prelude::*;

use std::cmp::PartialOrd;
use std::cell::RefCell;
use super::common::*;

pub fn refine_f0_stft_core(x: &[f32], in_out_f0_list: &mut [f32], hop_size: f64, fft_size: usize, win_b: f32, sr: f32, peak_search_range: f32) {
  let n_bin = fft_size / 2 + 1;

  struct Store {
    win_buffer: Vec<f32>,
    rfft_plan: R2CPlan32,
    fft_in: fftw::array::AlignedVec<f32>,
    fft_out: fftw::array::AlignedVec<Complex32>,
  }

  thread_local!(static STORE: RefCell<Option<Store>> = RefCell::new(None));
  in_out_f0_list.par_iter_mut().enumerate().for_each(|(i_hop, f0)| {
    STORE.with(|store_cell| {
      let mut store_option = store_cell.borrow_mut();
      if store_option.is_none() {
        store_option.replace(Store {
          win_buffer: vec![0.0f32; fft_size],
          rfft_plan: R2CPlan::aligned(&[fft_size], Flag::Estimate | Flag::DestroyInput).unwrap(),
          fft_in: fftw::array::AlignedVec::new(fft_size),
          fft_out: fftw::array::AlignedVec::new(n_bin),
        });
      }
      let Store { win_buffer, rfft_plan, fft_in, fft_out } = store_option.as_mut().unwrap();

      if *f0 <= 0.0 { return; }
      let i_center = (i_hop as f64 * hop_size).round() as usize;
      // stft
      let mut win_size = ((sr / *f0 * win_b * 2.0) as usize).min(fft_size);
      if win_size % 2 != 0 { win_size += 1; }
      let half_win_size = win_size / 2;

      let window = &mut win_buffer[..win_size];
      blackman(window);
      let (ob, oe, ib, ie) = get_frame_range(x.len() as isize, i_center as isize, win_size as isize);
      let dc = x[ib..ie].iter().sum::<f32>() / win_size as f32;
      fill(&mut window[..ob], 0.0);
      window[ob..oe].iter_mut().enumerate().for_each(|(i, v)| *v *= x[ib + i] - dc);
      fill(&mut window[oe..], 0.0);
      fft_in[..half_win_size].copy_from_slice(&window[half_win_size..]);
      fft_in[fft_size - half_win_size..].copy_from_slice(&window[..half_win_size]);
      fill(&mut fft_in[half_win_size..fft_size - half_win_size], 0.0);
      rfft_plan.r2c(fft_in, fft_out).unwrap();
      
      // find peak
      let lower_idx = ((*f0 * fft_size as f32 / sr * (1.0 - peak_search_range)) as usize).max(1);
      let upper_idx = ((*f0 * fft_size as f32 / sr * (1.0 + peak_search_range)) as usize).min(n_bin - 1);
      let peak_idx = fft_out[lower_idx..upper_idx].iter().map(|v| v.norm()).enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0 + lower_idx;

      // optimize peak
      let cost_function = |f| -calc_spectrum_at_freq(window, f, sr).norm().ln();
      let peak_freq = fmin_scalar(cost_function, ((peak_idx - 1) as f32 / fft_size as f32 * sr, (peak_idx + 1) as f32 / fft_size as f32 * sr), 32);
      let mut peak_phase = calc_spectrum_at_freq(window, peak_freq, sr).arg();
      peak_phase -= (peak_phase / 2.0 / std::f32::consts::PI).floor() * 2.0 * std::f32::consts::PI;

      // peak delta phase
      blackman(window);
      let (ob, oe, ib, ie) = get_frame_range(x.len() as isize, i_center as isize - 1, win_size as isize);
      let dc = x[ib..ie].iter().sum::<f32>() / win_size as f32;
      fill(&mut window[..ob], 0.0);
      window[ob..oe].iter_mut().enumerate().for_each(|(i, v)| *v *= x[ib + i] - dc);
      fill(&mut window[oe..], 0.0);
      let mut prev_peak_phase = calc_spectrum_at_freq(window, peak_freq, sr).arg();
      prev_peak_phase -= (prev_peak_phase / 2.0 / std::f32::consts::PI).floor() * 2.0 * std::f32::consts::PI;
      if peak_phase < prev_peak_phase { peak_phase += 2.0 * std::f32::consts::PI; }
      assert!(peak_phase >= prev_peak_phase);
      let refined_f0 = (peak_phase - prev_peak_phase) / 2.0 / std::f32::consts::PI * sr;
      if (refined_f0 - peak_freq).abs() > sr / fft_size as f32 * 1.5 { *f0 = peak_freq; }
      else { *f0 = refined_f0; }
    });
  });
}