use fftw::types::Flag;
use fftw::plan::{R2CPlan, C2RPlan, R2CPlan32, C2RPlan32};
use num_complex::*;
use rayon::prelude::*;

use std::cell::RefCell;
use super::common::*;

pub struct YinDifference<'a> {
  pub input: &'a [f32],
  pub rfft_plan: &'a mut R2CPlan32,
  pub irfft_plan: &'a mut C2RPlan32,
  pub fft_tmp_r: &'a mut [f32], // fft aligned, shape = (fft_size,)
  pub fft_tmp_a: &'a mut [Complex32], // fft aligned, shape = (fft_size / 2 + 1,)
  pub fft_tmp_b: &'a mut [Complex32], // fft aligned, shape = (fft_size / 2 + 1,)
  pub fft_size: usize, // greater than 0, power of 2
  pub out: &'a mut [f32],
}

impl <'a> YinDifference<'a> {
  pub fn exec(&mut self) {
    let in_size = self.input.len();
    let out_size = self.out.len();
    let n_bin = self.fft_size / 2 + 1;
    assert!(is_power_of_two(self.fft_size), "fft_size must be power of 2");
    assert_eq!(in_size % 2, 0, "input length must be even");
    assert_eq!(in_size / 2, out_size, "out_size must be in_size / 2");
    assert_eq!(self.fft_tmp_a.len(), n_bin);
    assert_eq!(self.fft_tmp_b.len(), n_bin);

    // calculate the power term, see yin paper eq. (7)
    // use self.out as power_terms
    let power_term_0 = self.input[..out_size].iter().map(|v| *v * *v).sum::<f32>();
    self.out[0] = power_term_0;
    for i in 1..out_size {
      self.out[i] = self.out[i - 1] - self.input[i - 1] * self.input[i - 1] + self.input[i + out_size] * self.input[i + out_size];
    }
    
    // Yin-style ACF via FFT
    // 1. data
    // use self.fft_tmp_a as transformed_audio
    self.fft_tmp_r[..in_size].copy_from_slice(self.input);
    fill(&mut self.fft_tmp_r[in_size..], 0.0);
    self.rfft_plan.r2c(self.fft_tmp_r, self.fft_tmp_a).unwrap();
    // 2. half of the data, disguised as a convolution kernel
    // use self.fft_tmp_b as transformed_kernel
    for i in 0..out_size {
      self.fft_tmp_r[i] = self.input[out_size - 1 - i];
    }
    fill(&mut self.fft_tmp_r[out_size..], 0.0);
    self.rfft_plan.r2c(self.fft_tmp_r, self.fft_tmp_b).unwrap();
    // 3. convolution
    // use self.fft_tmp_a as yin_style_acf
    for i in 0..n_bin {
      self.fft_tmp_a[i] *= self.fft_tmp_b[i];
    }
    // use self.fft_tmp_r as correlation
    self.irfft_plan.c2r(self.fft_tmp_a, self.fft_tmp_r).unwrap();
    
    // calculate difference function according to (7) in the Yin paper
    let fac = 2.0 / self.fft_size as f32; // for fftw, irfft(rfft(x)) == x.len() * x
    for i in 0..out_size {
      self.out[i] = power_term_0 + self.out[i] - fac * self.fft_tmp_r[out_size - 1 + i];
    }
  }
}

pub fn yin_cumulative_difference(in_out: &mut [f32]) {
  let n = in_out.len();
  
  in_out[0] = 1.0;
  let mut sum_value = 0.0;
  for i in 1..n {
    sum_value += in_out[i];
    if sum_value == 0.0 {
      in_out[i] = 1.0;
    }
    else {
      in_out[i] *= i as f32 / sum_value;
    }
  }
}

pub fn yin_find_valleys(x: &[f32], i_begin: usize, i_end: usize, mut threshold: f32, step: f32, out: &mut Vec<usize>, limit: usize) {
  assert!(i_begin > 0, "i_begin must be greater than 0");
  assert!(i_end < x.len(), "i_end must be less than x.len()");
  
  for i in i_begin..i_end {
    let prev = x[i - 1];
    let curr = x[i];
    let next = x[i + 1];
    if prev > curr && next > curr && curr < threshold {
      threshold = curr - step;
      out.push(i);
      if out.len() == limit {
        break;
      }
    }
  }
}

pub fn pyin_core(x: &[f32], pdf: &[f32], bias: f32, prob_threshold: f32, weight_prior: f32, hop_size: f64, valley_threshold: f32, valley_step: f32, min_freq: f32, max_freq: f32, min_win_size: f32, win_len_fac: f32, sr: f32, max_iter: usize, n_hop: usize, max_freq_prob_count: usize, remove_dc: bool, out: &mut [f32]) {
  let max_win_size = {
    let mut v = ((sr / min_freq * 4.0).max(min_win_size) * win_len_fac) as usize;
    if v % 2 == 1 { v += 1; }
    v
  };
  let half_max_win_size = max_win_size / 2;
  let fft_size = round_up_to_power_of_2(max_win_size);
  let n_bin = fft_size / 2 + 1;
  let pdf_size = pdf.len();

  assert!(max_win_size > 0);
  assert!(fft_size > 0);
  assert!(pdf_size > 0);
  assert!(min_freq >= 0.0 && max_freq <= sr && max_freq > min_freq);
  assert_eq!(out.len(), n_hop * max_freq_prob_count * 2);

  let rbounded_find_begin_idx = (sr / max_freq).max(1.0).min((half_max_win_size - 1) as f32) as usize;
  let rbounded_find_end_idx = (sr / min_freq).ceil().max(1.0).min((half_max_win_size - 1) as f32) as usize;
  
  struct Store {
    frame_buffer: Vec<f32>,
    yin_diff_buffer: Vec<f32>,
    rfft_plan: R2CPlan32,
    irfft_plan: C2RPlan32,
    fft_tmp_r: fftw::array::AlignedVec<f32>,
    fft_tmp_a: fftw::array::AlignedVec<Complex32>,
    fft_tmp_b: fftw::array::AlignedVec<Complex32>,
    valley_buffer: Vec<usize>,
  }

  fill(out, 0.0);
  thread_local!(static STORE: RefCell<Option<Store>> = RefCell::new(None));
  rayon::ThreadPoolBuilder::new().build().unwrap().install(move || {
    out.par_chunks_mut(max_freq_prob_count * 2).enumerate().for_each(|(i_hop, out_freq_prob_list)| {
      STORE.with(|store_cell| {
        let mut store_option = store_cell.borrow_mut();
        if store_option.is_none() {
          store_option.replace(Store {
            frame_buffer: vec![0.0f32; max_win_size],
            yin_diff_buffer: vec![0.0f32; max_win_size / 2],
            rfft_plan: R2CPlan::aligned(&[fft_size], Flag::Estimate | Flag::DestroyInput).unwrap(),
            irfft_plan: C2RPlan::aligned(&[fft_size], Flag::Estimate | Flag::DestroyInput).unwrap(),
            fft_tmp_r: fftw::array::AlignedVec::new(fft_size),
            fft_tmp_a: fftw::array::AlignedVec::new(n_bin),
            fft_tmp_b: fftw::array::AlignedVec::new(n_bin),
            valley_buffer: Vec::with_capacity(max_freq_prob_count),
          });
        }
        let Store { frame_buffer, yin_diff_buffer, rfft_plan, irfft_plan, fft_tmp_r, fft_tmp_a, fft_tmp_b, valley_buffer } = store_option.as_mut().unwrap();

        let center_idx = (i_hop as f64 * hop_size).round() as isize;
        let mut win_size = max_win_size;
        if win_size % 2 == 1 { win_size += 1; }
        // generate yin diff and find valleys
        valley_buffer.clear();
        for _ in 0..max_iter {
          let half_win_size = win_size / 2;
          let frame = &mut frame_buffer[..win_size];
          let yin_diff = &mut yin_diff_buffer[..half_win_size];
          get_frame(x, center_idx, frame);
          if remove_dc { do_remove_dc(frame); }
          YinDifference {
            input: frame,
            rfft_plan: rfft_plan,
            irfft_plan: irfft_plan,
            fft_tmp_r: fft_tmp_r,
            fft_tmp_a: fft_tmp_a,
            fft_tmp_b: fft_tmp_b,
            fft_size,
            out: yin_diff,
          }.exec();
          yin_cumulative_difference(yin_diff);
          
          valley_buffer.clear();
          let find_begin_idx = rbounded_find_begin_idx.min(half_win_size - 1);
          let find_end_idx = rbounded_find_end_idx.min(half_win_size - 1);
          yin_find_valleys(yin_diff, find_begin_idx, find_end_idx, valley_threshold, valley_step, valley_buffer, max_freq_prob_count);
          if valley_buffer.len() > 0 {
            let possible_freq = (sr / *valley_buffer.last().unwrap() as f32 - 20.0).max(min_freq).min(max_freq);
            let mut new_win_size = ((sr / possible_freq * 4.0).max(min_win_size) * win_len_fac) as usize;
            if new_win_size % 2 == 1 { new_win_size += 1; }
            if new_win_size == win_size { break; }
            win_size = new_win_size;
          }
        }
        // generate freq and prob
        let yin_diff = &yin_diff_buffer[..win_size / 2];
        let mut total_prob = 0.0;
        let mut total_weighted_prob = 0.0;
        for (i_valley, valley) in valley_buffer.iter().enumerate() {
          let (ipled_idx, ipled_val) = parabolic_interpolate(yin_diff, *valley, false);
          let freq = sr / ipled_idx;
          let v0 = if i_valley == 0 { 1.0 } else { (yin_diff[valley_buffer[i_valley - 1]].max(0.0) + EPS).min(1.0) };
          let v1 = if i_valley == valley_buffer.len() - 1 { 0.0 } else { (yin_diff[valley_buffer[i_valley + 1]].max(0.0) + EPS).min(1.0) };
          let mut prob = 0.0;
          for i in (v1 * pdf_size as f32) as usize..(v0 * pdf_size as f32) as usize {
            prob += pdf[i] * if ipled_val < i as f32 / pdf_size as f32 { 1.0 } else { 0.01 };
          }
          prob = prob.min(0.99) * bias;
          total_prob += prob;
          if ipled_val < prob_threshold { prob *= weight_prior; }
          total_weighted_prob += prob;
          out_freq_prob_list[i_valley * 2] = freq;
          out_freq_prob_list[i_valley * 2 + 1] = prob;
        }
        // renormalize
        if valley_buffer.len() > 0 && total_weighted_prob != 0.0 {
          let norm_fac = total_prob / total_weighted_prob;
          for i_valley in 0..valley_buffer.len() {
            out_freq_prob_list[i_valley * 2 + 1] *= norm_fac;
          }
        }
      });
    });
  });
}