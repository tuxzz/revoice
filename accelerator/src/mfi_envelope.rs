use super::common::*;

use num_complex::*;
use fftw::plan::{R2CPlan, R2CPlan32};
use fftw::types::Flag;
use rayon::prelude::*;

use std::cell::RefCell;

pub fn mfi_core(x: &[f32], f0_list: &[f32], kernel: &[f32], trans: &[f32], fixed_f0: f32, hop_size: f64, sr: f32, win_len_fac: f32, fft_size: usize, out: &mut [f32]) {
  let n_bin = fft_size / 2 + 1;
  let n_hop = f0_list.len();
  let krnl_size = kernel.len();
  let half_krnl_size = trans.len();
  let n_x = x.len();
  assert_eq!(out.len(), n_hop * n_bin, "bad out.len()");
  assert_eq!(krnl_size / 2, half_krnl_size, "bad krnl_size");

  struct Store {
    rfft_plan: R2CPlan32,
    fft_in: fftw::array::AlignedVec<f32>,
    fft_out: fftw::array::AlignedVec<Complex32>,
    window_temp: Vec<f32>,
    igd_magn: Vec<f32>,
    smoothed_magn: Vec<f32>,
  }

  thread_local!(static STORE: RefCell<Option<Store>> = RefCell::new(None));
  rayon::ThreadPoolBuilder::new().build().unwrap().install(move || {
    out.par_chunks_mut(n_bin).enumerate().for_each(move |(i_hop, out_bin_list)| {
      STORE.with(|store_cell| {
        let mut store_option = store_cell.borrow_mut();
        if store_option.is_none() {
          store_option.replace(Store {
            rfft_plan: R2CPlan::aligned(&[fft_size], Flag::Estimate | Flag::DestroyInput).unwrap(),
            fft_in: fftw::array::AlignedVec::new(fft_size),
            fft_out: fftw::array::AlignedVec::new(n_bin),
            window_temp: vec![0.0f32; fft_size],
            igd_magn: vec![0.0f32; n_bin],
            smoothed_magn: vec![0.0f32; n_bin],
          });
        }
        let Store { rfft_plan, fft_in, fft_out, window_temp, igd_magn, smoothed_magn } = store_option.as_mut().unwrap();

        let mut f0 = f0_list[i_hop];
        if f0 <= 0.0 { f0 = fixed_f0; }

        // generate window
        let i_center = (i_hop as f64 * hop_size).round() as isize;
        let offset_radius = (sr / (2.0 * f0)).ceil() as isize;
        let stdev = sr / (3.0 * f0);
        let mut window_size = ((2.0 * sr / f0 * win_len_fac) as usize).min(fft_size);
        if window_size % 2 != 0 { window_size += 1; }
        let window = &mut window_temp[..window_size];
        gaussian(stdev, window);
        let window_norm_fac = 2.0 / window.iter().sum::<f32>();
        window.iter_mut().for_each(|v| *v *= window_norm_fac);

        // calc average(integrated) magn
        let igd_magn_norm_fac = (2 * offset_radius) as f32;
        fill(igd_magn, 0.0);
        for offset in -offset_radius..offset_radius {
          let (ob, oe, ib, ie) = get_frame_range(n_x as isize, i_center + offset, window_size as isize);
          let dc = x[ib..ie].iter().sum::<f32>() / window_size as f32;
          fill(&mut fft_in[..ob], 0.0);
          for i in ob..oe { fft_in[i] = (x[ib + i - ob] - dc) * window[i]; }
          fill(&mut fft_in[oe..], 0.0);
          rfft_plan.r2c(fft_in, fft_out).unwrap();
          igd_magn.iter_mut().enumerate().for_each(|(i, v)| *v += fft_out[i].norm() / igd_magn_norm_fac);
        }
        let igd_energy = igd_magn.iter().map(|v| *v * *v).sum::<f32>();
        if igd_energy < 1e-16 {
          fill(out_bin_list, 1e-6);
          return;
        }

        // filter average magn on log domain
        igd_magn.iter_mut().for_each(|v| *v = v.max(1e-6).ln());
        convolve(&igd_magn, kernel, smoothed_magn, half_krnl_size);
        // make bounds better
        for i in 0..half_krnl_size {
          smoothed_magn[i] = igd_magn[i] + (smoothed_magn[i] - igd_magn[i]) * trans[i];
          smoothed_magn[n_bin - half_krnl_size + i] = igd_magn[n_bin - half_krnl_size + i] + (smoothed_magn[n_bin - half_krnl_size + i] - igd_magn[n_bin - half_krnl_size + i]) * trans[half_krnl_size - 1 - i];
        }
        // normalize filtered magn and output
        let norm_fac = (igd_energy / smoothed_magn.iter().map(|v| {
          let t = v.exp();
          t * t
        }).sum::<f32>()).sqrt().ln();
        out_bin_list.iter_mut().enumerate().for_each(|(i, v)| *v = smoothed_magn[i] + norm_fac);
      });
    });
  });
}