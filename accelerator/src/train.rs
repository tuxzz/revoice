use super::common::*;

use fftw::types::Flag;
use fftw::plan::{R2CPlan, R2CPlan32};
use num_traits::*;
use num_complex::*;
use rayon::prelude::*;
use rand::prelude::*;

use std::cell::RefCell;

thread_local! {
  static VUV_THREAD_POOL: RefCell<Option<rayon::ThreadPool>> = RefCell::new(None);
}

pub fn refresh_vuv_train() {
  VUV_THREAD_POOL.with(|vuv_thread_pool| {
    let mut thread_pool_option = vuv_thread_pool.borrow_mut();
    if thread_pool_option.is_some() { thread_pool_option.take(); }
  });
}
/*
pub fn calc_fft_gram(x: &[f32], window: &[f32], hop_size: f64, out_magn_list: &mut [f32]) {
  assert_eq!(window.len() % 2, 0);

  let fft_size = window.len();
  let half_win_size = fft_size / 2;
  let n_bin = fft_size / 2 + 1;
  let n_x = x.len();
  assert_eq!(out_magn_list.len() % (n_bin - 1), 0);

  struct Store {
    rfft_plan: R2CPlan32,
    fft_in: fftw::array::AlignedVec<f32>,
    fft_out: fftw::array::AlignedVec<Complex32>,
    frame_buffer: Vec<f32>,
  }
  thread_local!(static STORE: RefCell<Option<Store>> = RefCell::new(None));

  VUV_THREAD_POOL.with(|vuv_thread_pool| {
    let mut thread_pool_option = vuv_thread_pool.borrow_mut();
    let thread_pool = {
      if thread_pool_option.is_none() { thread_pool_option.replace(rayon::ThreadPoolBuilder::new().build().unwrap()); }
      thread_pool_option.as_ref().unwrap()
    };

    thread_pool.install(|| {
      out_magn_list.par_chunks_mut(n_bin - 1).enumerate().for_each(|(i_hop, out_magn)| {
        STORE.with(|store_cell| {
          let mut store_option = store_cell.borrow_mut();
          if store_option.is_none() {
            store_option.replace(Store {
              rfft_plan: R2CPlan::aligned(&[fft_size], Flag::Measure | Flag::DestroyInput).unwrap(),
              fft_in: fftw::array::AlignedVec::new(fft_size),
              fft_out: fftw::array::AlignedVec::new(n_bin),
              frame_buffer: vec![0.0; fft_size],
            });
          }
          let Store { rfft_plan, fft_in, fft_out, frame_buffer } = store_option.as_mut().unwrap();
          let i_center = (i_hop as f64 * hop_size).round() as isize;
          let (ob, oe, _, _) = get_frame_range(n_x as isize, i_center, fft_size as isize);
          fill(frame_buffer, 0.0);

          get_frame(x, i_center, frame_buffer);
          do_remove_dc(frame_buffer);
          for i_o in ob..oe { frame_buffer[i_o] *= window[i_o]; }

          for i in 0..half_win_size {
            fft_in[half_win_size + i] = frame_buffer[i];
            fft_in[i] = frame_buffer[half_win_size + i];
          }
          rfft_plan.r2c(fft_in, fft_out).unwrap();
          for i in 0..n_bin - 1 { out_magn[i] = (fft_out[i + 1].norm() + 1e-5).ln(); }
        });
      });
    });
  });
}

pub fn gen_vuv_train_sample(sinusoid: &[f32], noise: &[f32], window: &[f32], mean: &[f32], stdev: &[f32], hop_size: f64, out_magn_list: &mut [f32]) -> usize {
  assert_eq!(sinusoid.len(), noise.len());
  assert_eq!(mean.len(), stdev.len());
  assert_eq!(window.len() % 2, 0);

  let mut rng = rand::thread_rng();
  let fft_size = window.len();
  let half_win_size = fft_size / 2;
  let n_bin = fft_size / 2 + 1;
  let length = out_magn_list.len() / (n_bin - 1);
  let n_x = sinusoid.len();
  let n_hop = get_n_frame(n_x, hop_size);
  assert_eq!(out_magn_list.len() % (n_bin - 1), 0);
  assert_eq!(mean.len(), n_bin - 1);

  let (amp, amp_end, noise_ratio) = (rng.gen_range(0.05f32, 1.0f32), rng.gen_range(0.05f32, 1.0f32), rng.gen_range(0.0f32, 1.0f32));
  let i_begin_hop = rng.gen_range(0usize, n_hop - length);
  let use_fade = rng.gen::<bool>();
  
  let i_left = (i_begin_hop as f64 * hop_size).round() as isize - half_win_size as isize;
  let i_right = ((i_begin_hop + length) as f64 * hop_size).round() as isize + half_win_size as isize;
  let fade: Option<Vec<f32>> = if use_fade {
    let n = (i_right - i_left) as usize;
    Some((0..n).map(|i| amp + (amp_end - amp) * (i as f32 / (n - 1) as f32)).collect())
  }
  else { None };

  struct Store {
    rfft_plan: R2CPlan32,
    fft_in: fftw::array::AlignedVec<f32>,
    fft_out: fftw::array::AlignedVec<Complex32>,
    frame_buffer: Vec<f32>,
  }
  thread_local!(static STORE: RefCell<Option<Store>> = RefCell::new(None));

  VUV_THREAD_POOL.with(|vuv_thread_pool| {
    let mut thread_pool_option = vuv_thread_pool.borrow_mut();
    let thread_pool = {
      if thread_pool_option.is_none() { thread_pool_option.replace(rayon::ThreadPoolBuilder::new().build().unwrap()); }
      thread_pool_option.as_ref().unwrap()
    };

    thread_pool.install(|| {
      out_magn_list.par_chunks_mut(n_bin - 1).enumerate().for_each(|(i_hop, out_magn)| {
        STORE.with(|store_cell| {
          let mut store_option = store_cell.borrow_mut();
          if store_option.is_none() {
            store_option.replace(Store {
              rfft_plan: R2CPlan::aligned(&[fft_size], Flag::Measure | Flag::DestroyInput).unwrap(),
              fft_in: fftw::array::AlignedVec::new(fft_size),
              fft_out: fftw::array::AlignedVec::new(n_bin),
              frame_buffer: vec![0.0; fft_size],
            });
          }
          let Store { rfft_plan, fft_in, fft_out, frame_buffer } = store_option.as_mut().unwrap();
          let i_center = ((i_begin_hop + i_hop) as f64 * hop_size).round() as isize;
          let (ob, oe, ib, _) = get_frame_range(n_x as isize, i_center, fft_size as isize);
          let (_, _, ib_fade, _) = get_frame_range(i_right - i_left, i_center - i_left, fft_size as isize);
          fill(frame_buffer, 0.0);

          match fade.as_ref() {
            Some(fade_fac) => {
              for i_o in ob..oe {
                let i_i = ib + i_o - ob;
                let sinusoid_sample = sinusoid[i_i];
                let noise_sample = noise[i_i];
                let sample = (sinusoid_sample + noise_sample * noise_ratio) * fade_fac[ib_fade];
                frame_buffer[i_o] = sample;
              }
            },
            None => {
              for i_o in ob..oe {
                let i_i = ib + i_o - ob;
                let sinusoid_sample = sinusoid[i_i];
                let noise_sample = noise[i_i];
                let sample = (sinusoid_sample + noise_sample * noise_ratio) * amp;
                frame_buffer[i_o] = sample;
              }
            }
          }
          do_remove_dc(frame_buffer);
          for i_o in ob..oe { frame_buffer[i_o] *= window[i_o]; }

          for i in 0..half_win_size {
            fft_in[half_win_size + i] = frame_buffer[i];
            fft_in[i] = frame_buffer[half_win_size + i];
          }
          rfft_plan.r2c(fft_in, fft_out).unwrap();
          for i in 0..n_bin - 1 { out_magn[i] = ((fft_out[i + 1].norm() + 1e-5).ln() - mean[i]) / stdev[i]; }
        });
      });
    });
  });

  i_begin_hop
}
*/

pub fn calc_spectrum(x: &[f32], freq_list: &[f32], window: &[f32], hop_size: f64, out_magn_list: &mut [f32], fft_size: usize, sr: f32) {
  assert_eq!(window.len() % 2, 0);

  let win_size = window.len();
  let n_bin = freq_list.len();
  let n_fft_bin = fft_size / 2 + 1;
  let n_x = x.len();
  assert_eq!(out_magn_list.len() % n_bin, 0);

  struct Store {
    rfft_plan: R2CPlan32,
    fft_in: fftw::array::AlignedVec<f32>,
    fft_out: fftw::array::AlignedVec<Complex32>,
  }
  thread_local!(static STORE: RefCell<Option<Store>> = RefCell::new(None));

  VUV_THREAD_POOL.with(|vuv_thread_pool| {
    let mut thread_pool_option = vuv_thread_pool.borrow_mut();
    let thread_pool = {
      if thread_pool_option.is_none() { thread_pool_option.replace(rayon::ThreadPoolBuilder::new().build().unwrap()); }
      thread_pool_option.as_ref().unwrap()
    };

    thread_pool.install(|| {
      out_magn_list.par_chunks_mut(n_bin).enumerate().for_each(|(i_hop, out_magn)| {
        STORE.with(|store_cell| {
          let mut store_option = store_cell.borrow_mut();
          if store_option.is_none() {
            store_option.replace(Store {
              rfft_plan: R2CPlan::aligned(&[fft_size], Flag::Measure | Flag::DestroyInput).unwrap(),
              fft_in: fftw::array::AlignedVec::new(fft_size),
              fft_out: fftw::array::AlignedVec::new(n_fft_bin),
            });
          }
          let Store { rfft_plan, fft_in, fft_out } = store_option.as_mut().unwrap();
          let i_center = (i_hop as f64 * hop_size).round() as isize;
          let (ob, oe, _, _) = get_frame_range(n_x as isize, i_center, win_size as isize);
          fill(fft_in, 0.0);
          {
            let frame_buffer = &mut fft_in[..win_size];
            get_frame(x, i_center, frame_buffer);
            do_remove_dc(&mut frame_buffer[ob..oe]);
            for i_o in ob..oe { frame_buffer[i_o] *= window[i_o]; }
          }
          rfft_plan.r2c(fft_in, fft_out).unwrap();
          freq_list.iter().enumerate().for_each(|(i, freq)| {
            let i_fft_bin = (((freq / sr) * fft_size as f32).round().max(0.0) as usize).min(n_fft_bin);
            out_magn[i] = (fft_out[i_fft_bin].norm() + 1e-5).ln()
          });
        });
      });
    });
  });
}

pub fn gen_vuv_train_sample(sinusoid: &[f32], noise: &[f32], freq_list: &[f32], window: &[f32], mean: &[f32], stdev: &[f32], hop_size: f64, out_magn_list: &mut [f32], fft_size: usize, sr: f32) -> usize {
  assert_eq!(sinusoid.len(), noise.len());
  assert_eq!(mean.len(), stdev.len());
  assert_eq!(window.len() % 2, 0);

  let mut rng = rand::thread_rng();
  let win_size = window.len();
  let half_win_size = win_size / 2;
  let n_fft_bin = fft_size / 2 + 1;
  let n_bin = freq_list.len();
  let length = out_magn_list.len() / n_bin;
  let n_x = sinusoid.len();
  let n_hop = get_n_frame(n_x, hop_size);
  assert_eq!(out_magn_list.len() % n_bin, 0);
  assert_eq!(mean.len(), n_bin);

  let (amp, amp_end, noise_ratio) = (rng.gen_range(0.05f32, 1.0f32), rng.gen_range(0.05f32, 1.0f32), rng.gen_range(0.0f32, 1.0f32));
  let i_begin_hop = rng.gen_range(0usize, n_hop - length);
  let use_fade = rng.gen::<bool>();
  
  let i_left = (i_begin_hop as f64 * hop_size).round() as isize - half_win_size as isize;
  let i_right = ((i_begin_hop + length) as f64 * hop_size).round() as isize + half_win_size as isize;
  let fade: Option<Vec<f32>> = if use_fade {
    let n = (i_right - i_left) as usize;
    Some((0..n).map(|i| amp + (amp_end - amp) * (i as f32 / (n - 1) as f32)).collect())
  }
  else { None };

  struct Store {
    rfft_plan: R2CPlan32,
    fft_in: fftw::array::AlignedVec<f32>,
    fft_out: fftw::array::AlignedVec<Complex32>,
  }
  thread_local!(static STORE: RefCell<Option<Store>> = RefCell::new(None));

  VUV_THREAD_POOL.with(|vuv_thread_pool| {
    let mut thread_pool_option = vuv_thread_pool.borrow_mut();
    let thread_pool = {
      if thread_pool_option.is_none() { thread_pool_option.replace(rayon::ThreadPoolBuilder::new().build().unwrap()); }
      thread_pool_option.as_ref().unwrap()
    };

    thread_pool.install(|| {
      out_magn_list.par_chunks_mut(n_bin).enumerate().for_each(|(i_hop, out_magn)| {
        STORE.with(|store_cell| {
          let mut store_option = store_cell.borrow_mut();
          if store_option.is_none() {
            store_option.replace(Store {
              rfft_plan: R2CPlan::aligned(&[fft_size], Flag::Measure | Flag::DestroyInput).unwrap(),
              fft_in: fftw::array::AlignedVec::new(fft_size),
              fft_out: fftw::array::AlignedVec::new(n_fft_bin),
            });
          }
          let Store { rfft_plan, fft_in, fft_out } = store_option.as_mut().unwrap();
          let i_center = ((i_begin_hop + i_hop) as f64 * hop_size).round() as isize;
          let (ob, oe, ib, _) = get_frame_range(n_x as isize, i_center, win_size as isize);
          let (_, _, ib_fade, _) = get_frame_range(i_right - i_left, i_center - i_left, win_size as isize);
          fill(fft_in, 0.0);

          {
            let frame_buffer = &mut fft_in[..win_size];
            match fade.as_ref() {
              Some(fade_fac) => {
                for i_o in ob..oe {
                  let i_i = ib + i_o - ob;
                  let sinusoid_sample = sinusoid[i_i];
                  let noise_sample = noise[i_i];
                  let sample = (sinusoid_sample + noise_sample * noise_ratio) * fade_fac[ib_fade];
                  frame_buffer[i_o] = sample;
                }
              },
              None => {
                for i_o in ob..oe {
                  let i_i = ib + i_o - ob;
                  let sinusoid_sample = sinusoid[i_i];
                  let noise_sample = noise[i_i];
                  let sample = (sinusoid_sample + noise_sample * noise_ratio) * amp;
                  frame_buffer[i_o] = sample;
                }
              }
            }
            do_remove_dc(&mut frame_buffer[ob..oe]);
            for i_o in ob..oe { frame_buffer[i_o] *= window[i_o]; }
          }
          rfft_plan.r2c(fft_in, fft_out).unwrap();
          freq_list.iter().enumerate().for_each(|(i, freq)| {
            let i_fft_bin = (((freq / sr) * fft_size as f32).round().max(0.0) as usize).min(n_fft_bin);
            out_magn[i] = ((fft_out[i_fft_bin].norm() + 1e-5).ln() - mean[i]) / stdev[i];
          });
        });
      });
    });
  });

  i_begin_hop
}