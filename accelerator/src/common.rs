use num_traits::*;
use num_complex::*;
use std::cmp::PartialOrd;
use std::sync::Mutex;
use std::cell::RefCell;
use rayon::prelude::*;
use fftw::types::Sign;

use fftw::types::Flag;
use fftw::plan::{R2CPlan, C2RPlan, C2CPlan, R2CPlan32, C2RPlan32, C2CPlan32};

pub const EPS: f32 = 2.2204e-16;

pub fn get_frame_range(input_len: isize, center: isize, size: isize) -> (usize, usize, usize, usize) {
  let left_size = size / 2;
  let right_size = size - left_size; // for odd size

  let input_begin = (center - left_size).max(0).min(input_len) as usize;
  let input_end = (center + right_size).max(0).min(input_len) as usize;

  let out_begin = (left_size - center).max(0) as usize;
  let out_end = out_begin + (input_end - input_begin);

  (out_begin, out_end, input_begin, input_end)
}

pub fn get_frame(input: &[f32], center: isize, out: &mut [f32]) {
  let (out_begin, out_end, input_begin, input_end) = get_frame_range(input.len() as isize, center, out.len() as isize);
  fill(&mut out[..out_begin], 0.0);
  out[out_begin..out_end].copy_from_slice(&input[input_begin..input_end]);
  fill(&mut out[out_end..], 0.0);
}

pub fn get_n_frame(input_size: usize, hop_size: f64) -> usize {
  assert!(hop_size > 0.0);
  (input_size as f64 / hop_size).round() as usize
}

pub fn gaussian(std: f32, out: &mut [f32]) {
  let sig2 = 2.0 * std * std;
  let m = out.len();
  let k = (m - 1) as f32 / 2.0;
  out.iter_mut().enumerate().for_each(|(i, v)| {
    let n = i as f32 - k;
    *v = (-(n * n) / sig2).exp();
  });
}

pub fn blackman(out: &mut [f32]) {
  if out.len() == 1 { out[0] = 1.0; }
  else {
    let d = (out.len() - 1) as f32;
    const PI: f32 = std::f32::consts::PI;
    out.iter_mut().enumerate().for_each(|(i, v)| {
      let k = i as f32;
      *v = 0.42 - 0.5 * (2.0 * PI * k / d).cos() + 0.08 * (4.0 * PI * k / d).cos();
    });
  }
}

pub fn convolve(x: &[f32], y: &[f32], out: &mut [f32], n_skip: usize) {
  let n_x = x.len();
  let n_y = y.len();
  assert!(n_x > 0 && n_y > 0, "length x and y must be greater than 0");
  assert!(out.len() <= n_x + n_y - 1 - n_skip);
  out.iter_mut().enumerate().for_each(|(raw_i, v)| {
    let i = raw_i + n_skip;
    let mut s = 0.0;
    let j_begin = (i as isize - n_x as isize + 1).max(0) as usize;
    let j_end = (i + 1).min(n_y);
    for j in j_begin..j_end {
      s += y[j] * x[i - j];
    }
    *v = s;
  });
}

pub fn do_remove_dc(x: &mut [f32]) {
  if x.len() == 0 { return; }
  let mean = x.iter().sum::<f32>() / x.len() as f32;
  x.iter_mut().for_each(|v| *v -= mean);
}

pub fn calc_spectrum_at_freq(x: &[f32], freq: f32, sr: f32) -> Complex32 {
  assert_eq!(x.len() % 2, 0);
  let half_n_x = (x.len() / 2) as isize;
  x.iter().enumerate().map(|(i, v)| {
    let t = (i as isize - half_n_x) as f32 / sr;
    let n2jpitf = Complex32::new(0.0, -2.0 * std::f32::consts::PI * t * freq);
    *v * n2jpitf.exp()
  }).sum::<Complex32>()
}

pub fn parabolic_interpolate(x: &[f32], i: usize, over_adjust: bool) -> (f32, f32) {
  let n = x.len();
  if i > 0 && i < (n - 1) {
    let s0 = x[i - 1];
    let s1 = x[i];
    let s2 = x[i + 1];
    let a = (s0 + s2) / 2.0 - s1;
    if a == 0.0 { return (i as f32, x[i]); }
    let b = s2 - s1 - a;
    let adjustment = -(b / a * 0.5);
    if !over_adjust && adjustment > 1.0 { return (i as f32, x[i]); }
    let x = i as f32 + adjustment;
    let y = a * adjustment * adjustment + b * adjustment + s1;
    return (x, y);
  }
  else {
    let adjusted_i = i.max(0).min(n - 1);
    return (adjusted_i as f32, x[adjusted_i]);
  }
}

pub fn fill<T: Copy>(x: &mut [T], value: T) {
  x.iter_mut().for_each(|v| *v = value);
}

pub fn is_power_of_two<T: Unsigned + PrimInt>(x: T) -> bool {
  !x.is_zero() && (x & (x - T::one())).is_zero()
}
pub fn round_up_to_power_of_2<T: Unsigned + PrimInt>(mut x: T) -> T {
  if x.is_zero() { return x; }
  x = x - T::one();
  let n_bit = T::max_value().count_ones() as usize;
  let mut i = 1usize;
  while i < n_bit {
    x = x | (x >> i);
    i *= 2;
  }
  x + T::one()
}

pub fn fmin_scalar<T, U, F>(f: F, bound_list: (T, T), max_iter: usize) -> T
where
  T: Num + Copy,
  U: Num + PartialOrd,
  F: Fn(T) -> U,
{
  let two = T::one() + T::one();
  let (mut x_left, mut x_right) = bound_list;
  let (mut f_left, mut f_right) = (f(x_left), f(x_right));
  for _ in 0..max_iter {
    if x_left == x_right { break; }
    let x_mid = (x_left + x_right) / two;
    let f_mid = f(x_mid);
    if f_left < f_right { x_right = x_mid; f_right = f_mid; }
    else { x_left = x_mid; f_left = f_mid; }
  }
  if f_left < f_right { x_left } else { x_right }
}

/* segconv mt */
pub fn segconv(x: &[f32], k: &[f32], out: &mut [f32]) {
  let n_x = x.len();
  let n_k = k.len();
  assert!(n_k >= 1);
  assert!(n_x >= n_k);
  assert_eq!(n_k % 2, 0);
  let n_out = n_x + n_k - 1;
  assert_eq!(out.len(), n_out);
  let fft_size = round_up_to_power_of_2(n_k * 2 - 1);
  let n_bin = fft_size / 2 + 1;
  // prepare kernel
  let fd_krnl_own = {
    let mut rfft_plan: R2CPlan32 = R2CPlan::aligned(&[fft_size], Flag::Estimate | Flag::DestroyInput).unwrap();
    let mut conv_io = fftw::array::AlignedVec::new(fft_size);
    let mut fft_tmp = fftw::array::AlignedVec::new(n_bin);
    conv_io[..n_k].copy_from_slice(k);
    fill(&mut conv_io[n_k..], 0.0);
    rfft_plan.r2c(&mut conv_io, &mut fft_tmp).unwrap();
    fft_tmp.iter_mut().for_each(|v| *v /= fft_size as f32);
    fft_tmp
  };
  let fd_krnl = fd_krnl_own.as_slice();

  // main
  struct Store {
    rfft_plan: R2CPlan32,
    irfft_plan: C2RPlan32,
    conv_io: fftw::array::AlignedVec<f32>,
    fft_tmp: fftw::array::AlignedVec<Complex32>,
  }
  thread_local!(static STORE: RefCell<Option<Store>> = RefCell::new(None));
  
  let chunk_size = fft_size - n_k + 1;
  let n_chunk = (n_x as f64 / chunk_size as f64).ceil() as usize;
  let out_lock = Mutex::new(out);
  rayon::ThreadPoolBuilder::new().build().unwrap().install(move || {
    (0..n_chunk).into_par_iter().for_each(|i_chunk| {
      STORE.with(|store_cell| {
        let mut store_option = store_cell.borrow_mut();
        if store_option.is_none() {
          store_option.replace(Store {
            rfft_plan: R2CPlan::aligned(&[fft_size], Flag::Estimate | Flag::DestroyInput).unwrap(),
            irfft_plan: C2RPlan::aligned(&[fft_size], Flag::Estimate | Flag::DestroyInput).unwrap(),
            conv_io: fftw::array::AlignedVec::new(fft_size),
            fft_tmp: fftw::array::AlignedVec::new(n_bin),
          });
        }
        let Store { rfft_plan, irfft_plan, conv_io, fft_tmp } = store_option.as_mut().unwrap();

        let in_begin = i_chunk * chunk_size;
        let in_end = ((i_chunk + 1) * chunk_size).min(n_x);
        assert!(in_end <= n_x);
        assert!(in_begin < in_end);
        let real_chunk_size = in_end - in_begin;
        conv_io[..real_chunk_size].copy_from_slice(&x[in_begin..in_end]);
        fill(&mut conv_io[real_chunk_size..], 0.0);
        rfft_plan.r2c(conv_io, fft_tmp).unwrap();
        fft_tmp.iter_mut().zip(fd_krnl.iter()).for_each(|(v, k)| *v *= *k);
        irfft_plan.c2r(fft_tmp, conv_io).unwrap();
        {
          let mut out = out_lock.lock().unwrap();
          out[in_begin..in_begin + real_chunk_size + n_k - 1].iter_mut().zip(conv_io[..real_chunk_size + n_k - 1].iter()).for_each(|(x, v)| *x += *v);
        }
      });
    });
  });
}

pub fn segconv_cplx(x: &[Complex32], k: &[Complex32], out: &mut [Complex32])
{
  let n_x = x.len();
  let n_k = k.len();
  assert!(n_k >= 1);
  assert!(n_x >= n_k);
  assert_eq!(n_k % 2, 0);
  let n_out = n_x + n_k - 1;
  assert_eq!(out.len(), n_out);
  let fft_size = round_up_to_power_of_2(n_k * 2 - 1);
  // prepare kernel
  let fd_krnl_own = {
    let mut fft_plan: C2CPlan32 = C2CPlan::aligned(&[fft_size], Sign::Forward, Flag::Estimate | Flag::DestroyInput).unwrap();
    let mut conv_io = fftw::array::AlignedVec::new(fft_size);
    let mut fft_tmp = fftw::array::AlignedVec::new(fft_size);
    conv_io[..n_k].copy_from_slice(k);
    fill(&mut conv_io[n_k..], Complex32::zero());
    fft_plan.c2c(&mut conv_io, &mut fft_tmp).unwrap();
    fft_tmp.iter_mut().for_each(|v| *v /= fft_size as f32);
    fft_tmp
  };
  let fd_krnl = fd_krnl_own.as_slice();

  // main
  struct Store {
    fft_plan: C2CPlan32,
    ifft_plan: C2CPlan32,
    conv_io: fftw::array::AlignedVec<Complex32>,
    fft_tmp: fftw::array::AlignedVec<Complex32>,
  }
  thread_local!(static STORE: RefCell<Option<Store>> = RefCell::new(None));
  
  let chunk_size = fft_size - n_k + 1;
  let n_chunk = (n_x as f64 / chunk_size as f64).ceil() as usize;
  let out_lock = Mutex::new(out);
  rayon::ThreadPoolBuilder::new().build().unwrap().install(move || {
    (0..n_chunk).into_par_iter().for_each(|i_chunk| {
      STORE.with(|store_cell| {
        let mut store_option = store_cell.borrow_mut();
        if store_option.is_none() {
          store_option.replace(Store {
            fft_plan: C2CPlan::aligned(&[fft_size], Sign::Forward, Flag::Estimate | Flag::DestroyInput).unwrap(),
            ifft_plan: C2CPlan::aligned(&[fft_size], Sign::Backward, Flag::Estimate | Flag::DestroyInput).unwrap(),
            conv_io: fftw::array::AlignedVec::new(fft_size),
            fft_tmp: fftw::array::AlignedVec::new(fft_size),
          });
        }
        let Store { fft_plan, ifft_plan, conv_io, fft_tmp } = store_option.as_mut().unwrap();

        let in_begin = i_chunk * chunk_size;
        let in_end = ((i_chunk + 1) * chunk_size).min(n_x);
        assert!(in_end <= n_x);
        assert!(in_begin < in_end);
        let real_chunk_size = in_end - in_begin;
        conv_io[..real_chunk_size].copy_from_slice(&x[in_begin..in_end]);
        fill(&mut conv_io[real_chunk_size..], Complex32::zero());
        fft_plan.c2c(conv_io, fft_tmp).unwrap();
        fft_tmp.iter_mut().zip(fd_krnl.iter()).for_each(|(v, k)| *v *= *k);
        ifft_plan.c2c(fft_tmp, conv_io).unwrap();
        {
          let mut out = out_lock.lock().unwrap();
          out[in_begin..in_begin + real_chunk_size + n_k - 1].iter_mut().zip(conv_io[..real_chunk_size + n_k - 1].iter()).for_each(|(x, v)| *x += *v);
        }
      });
    });
  });
}

pub fn segconv_real_cplx(x: &[f32], k: &[Complex32], out: &mut [Complex32])
{
  let n_x = x.len();
  let n_k = k.len();
  assert!(n_k >= 1);
  assert!(n_x >= n_k);
  assert_eq!(n_k % 2, 0);
  let n_out = n_x + n_k - 1;
  assert_eq!(out.len(), n_out);
  let fft_size = round_up_to_power_of_2(n_k * 2 - 1);
  // prepare kernel
  let fd_krnl_own = {
    let mut fft_plan: C2CPlan32 = C2CPlan::aligned(&[fft_size], Sign::Forward, Flag::Estimate | Flag::DestroyInput).unwrap();
    let mut conv_io = fftw::array::AlignedVec::new(fft_size);
    let mut fft_tmp = fftw::array::AlignedVec::new(fft_size);
    conv_io[..n_k].copy_from_slice(k);
    fill(&mut conv_io[n_k..], Complex32::zero());
    fft_plan.c2c(&mut conv_io, &mut fft_tmp).unwrap();
    fft_tmp.iter_mut().for_each(|v| *v /= fft_size as f32);
    fft_tmp
  };
  let fd_krnl = fd_krnl_own.as_slice();

  // main
  struct Store {
    fft_plan: C2CPlan32,
    ifft_plan: C2CPlan32,
    conv_io: fftw::array::AlignedVec<Complex32>,
    fft_tmp: fftw::array::AlignedVec<Complex32>,
  }
  thread_local!(static STORE: RefCell<Option<Store>> = RefCell::new(None));
  
  let chunk_size = fft_size - n_k + 1;
  let n_chunk = (n_x as f64 / chunk_size as f64).ceil() as usize;
  let out_lock = Mutex::new(out);
  rayon::ThreadPoolBuilder::new().build().unwrap().install(move || {
    (0..n_chunk).into_par_iter().for_each(|i_chunk| {
      STORE.with(|store_cell| {
        let mut store_option = store_cell.borrow_mut();
        if store_option.is_none() {
          store_option.replace(Store {
            fft_plan: C2CPlan::aligned(&[fft_size], Sign::Forward, Flag::Estimate | Flag::DestroyInput).unwrap(),
            ifft_plan: C2CPlan::aligned(&[fft_size], Sign::Backward, Flag::Estimate | Flag::DestroyInput).unwrap(),
            conv_io: fftw::array::AlignedVec::new(fft_size),
            fft_tmp: fftw::array::AlignedVec::new(fft_size),
          });
        }
        let Store { fft_plan, ifft_plan, conv_io, fft_tmp } = store_option.as_mut().unwrap();

        let in_begin = i_chunk * chunk_size;
        let in_end = ((i_chunk + 1) * chunk_size).min(n_x);
        assert!(in_end <= n_x);
        assert!(in_begin < in_end);
        let real_chunk_size = in_end - in_begin;
        conv_io[..real_chunk_size].iter_mut().zip(x[in_begin..in_end].iter()).for_each(|(a, b)| *a = Complex32::new(*b, 0.0));
        fill(&mut conv_io[real_chunk_size..], Complex32::zero());
        fft_plan.c2c(conv_io, fft_tmp).unwrap();
        fft_tmp.iter_mut().zip(fd_krnl.iter()).for_each(|(v, k)| *v *= *k);
        ifft_plan.c2c(fft_tmp, conv_io).unwrap();
        {
          let mut out = out_lock.lock().unwrap();
          out[in_begin..in_begin + real_chunk_size + n_k - 1].iter_mut().zip(conv_io[..real_chunk_size + n_k - 1].iter()).for_each(|(x, v)| *x += *v);
        }
      });
    });
  });
}

/* segconv st */
pub fn segconv_st(x: &[f32], k: &[f32], out: &mut [f32]) {
  let n_x = x.len();
  let n_k = k.len();
  assert!(n_k >= 1);
  assert!(n_x >= n_k);
  assert_eq!(n_k % 2, 0);
  let n_out = n_x + n_k - 1;
  assert_eq!(out.len(), n_out);
  let fft_size = round_up_to_power_of_2(n_k * 2 - 1);
  let n_bin = fft_size / 2 + 1;
  // prepare kernel
  let fd_krnl_own = {
    let mut rfft_plan: R2CPlan32 = R2CPlan::aligned(&[fft_size], Flag::Estimate | Flag::DestroyInput).unwrap();
    let mut conv_io = fftw::array::AlignedVec::new(fft_size);
    let mut fft_tmp = fftw::array::AlignedVec::new(n_bin);
    conv_io[..n_k].copy_from_slice(k);
    fill(&mut conv_io[n_k..], 0.0);
    rfft_plan.r2c(&mut conv_io, &mut fft_tmp).unwrap();
    fft_tmp.iter_mut().for_each(|v| *v /= fft_size as f32);
    fft_tmp
  };
  let fd_krnl = fd_krnl_own.as_slice();

  // main
  struct Store {
    rfft_plan: R2CPlan32,
    irfft_plan: C2RPlan32,
    conv_io: fftw::array::AlignedVec<f32>,
    fft_tmp: fftw::array::AlignedVec<Complex32>,
  }
  let mut store = Store {
    rfft_plan: R2CPlan::aligned(&[fft_size], Flag::Estimate | Flag::DestroyInput).unwrap(),
    irfft_plan: C2RPlan::aligned(&[fft_size], Flag::Estimate | Flag::DestroyInput).unwrap(),
    conv_io: fftw::array::AlignedVec::new(fft_size),
    fft_tmp: fftw::array::AlignedVec::new(n_bin),
  };
  
  let chunk_size = fft_size - n_k + 1;
  let n_chunk = (n_x as f64 / chunk_size as f64).ceil() as usize;
  (0..n_chunk).for_each(|i_chunk| {
    let Store { rfft_plan, irfft_plan, conv_io, fft_tmp } = &mut store;

    let in_begin = i_chunk * chunk_size;
    let in_end = ((i_chunk + 1) * chunk_size).min(n_x);
    assert!(in_end <= n_x);
    assert!(in_begin < in_end);
    let real_chunk_size = in_end - in_begin;
    conv_io[..real_chunk_size].copy_from_slice(&x[in_begin..in_end]);
    fill(&mut conv_io[real_chunk_size..], 0.0);
    rfft_plan.r2c(conv_io, fft_tmp).unwrap();
    fft_tmp.iter_mut().zip(fd_krnl.iter()).for_each(|(v, k)| *v *= *k);
    irfft_plan.c2r(fft_tmp, conv_io).unwrap();
    out[in_begin..in_begin + real_chunk_size + n_k - 1].iter_mut().zip(conv_io[..real_chunk_size + n_k - 1].iter()).for_each(|(x, v)| *x += *v);
  });
}

pub fn segconv_cplx_st(x: &[Complex32], k: &[Complex32], out: &mut [Complex32])
{
  let n_x = x.len();
  let n_k = k.len();
  assert!(n_k >= 1);
  assert!(n_x >= n_k);
  assert_eq!(n_k % 2, 0);
  let n_out = n_x + n_k - 1;
  assert_eq!(out.len(), n_out);
  let fft_size = round_up_to_power_of_2(n_k * 2 - 1);
  // prepare kernel
  let fd_krnl_own = {
    let mut fft_plan: C2CPlan32 = C2CPlan::aligned(&[fft_size], Sign::Forward, Flag::Estimate | Flag::DestroyInput).unwrap();
    let mut conv_io = fftw::array::AlignedVec::new(fft_size);
    let mut fft_tmp = fftw::array::AlignedVec::new(fft_size);
    conv_io[..n_k].copy_from_slice(k);
    fill(&mut conv_io[n_k..], Complex32::zero());
    fft_plan.c2c(&mut conv_io, &mut fft_tmp).unwrap();
    fft_tmp.iter_mut().for_each(|v| *v /= fft_size as f32);
    fft_tmp
  };
  let fd_krnl = fd_krnl_own.as_slice();

  // main
  struct Store {
    fft_plan: C2CPlan32,
    ifft_plan: C2CPlan32,
    conv_io: fftw::array::AlignedVec<Complex32>,
    fft_tmp: fftw::array::AlignedVec<Complex32>,
  }
  let mut store = Store {
    fft_plan: C2CPlan::aligned(&[fft_size], Sign::Forward, Flag::Estimate | Flag::DestroyInput).unwrap(),
    ifft_plan: C2CPlan::aligned(&[fft_size], Sign::Backward, Flag::Estimate | Flag::DestroyInput).unwrap(),
    conv_io: fftw::array::AlignedVec::new(fft_size),
    fft_tmp: fftw::array::AlignedVec::new(fft_size),
  };
  
  let chunk_size = fft_size - n_k + 1;
  let n_chunk = (n_x as f64 / chunk_size as f64).ceil() as usize;
  (0..n_chunk).for_each(|i_chunk| {
    let Store { fft_plan, ifft_plan, conv_io, fft_tmp } = &mut store;

    let in_begin = i_chunk * chunk_size;
    let in_end = ((i_chunk + 1) * chunk_size).min(n_x);
    assert!(in_end <= n_x);
    assert!(in_begin < in_end);
    let real_chunk_size = in_end - in_begin;
    conv_io[..real_chunk_size].copy_from_slice(&x[in_begin..in_end]);
    fill(&mut conv_io[real_chunk_size..], Complex32::zero());
    fft_plan.c2c(conv_io, fft_tmp).unwrap();
    fft_tmp.iter_mut().zip(fd_krnl.iter()).for_each(|(v, k)| *v *= *k);
    ifft_plan.c2c(fft_tmp, conv_io).unwrap();
    out[in_begin..in_begin + real_chunk_size + n_k - 1].iter_mut().zip(conv_io[..real_chunk_size + n_k - 1].iter()).for_each(|(x, v)| *x += *v);
  });
}

pub fn segconv_real_cplx_st(x: &[f32], k: &[Complex32], out: &mut [Complex32])
{
  let n_x = x.len();
  let n_k = k.len();
  assert!(n_k >= 1);
  assert!(n_x >= n_k);
  assert_eq!(n_k % 2, 0);
  let n_out = n_x + n_k - 1;
  assert_eq!(out.len(), n_out);
  let fft_size = round_up_to_power_of_2(n_k * 2 - 1);
  // prepare kernel
  let fd_krnl_own = {
    let mut fft_plan: C2CPlan32 = C2CPlan::aligned(&[fft_size], Sign::Forward, Flag::Estimate | Flag::DestroyInput).unwrap();
    let mut conv_io = fftw::array::AlignedVec::new(fft_size);
    let mut fft_tmp = fftw::array::AlignedVec::new(fft_size);
    conv_io[..n_k].copy_from_slice(k);
    fill(&mut conv_io[n_k..], Complex32::zero());
    fft_plan.c2c(&mut conv_io, &mut fft_tmp).unwrap();
    fft_tmp.iter_mut().for_each(|v| *v /= fft_size as f32);
    fft_tmp
  };
  let fd_krnl = fd_krnl_own.as_slice();

  // main
  struct Store {
    fft_plan: C2CPlan32,
    ifft_plan: C2CPlan32,
    conv_io: fftw::array::AlignedVec<Complex32>,
    fft_tmp: fftw::array::AlignedVec<Complex32>,
  }
  let mut store = Store {
    fft_plan: C2CPlan::aligned(&[fft_size], Sign::Forward, Flag::Estimate | Flag::DestroyInput).unwrap(),
    ifft_plan: C2CPlan::aligned(&[fft_size], Sign::Backward, Flag::Estimate | Flag::DestroyInput).unwrap(),
    conv_io: fftw::array::AlignedVec::new(fft_size),
    fft_tmp: fftw::array::AlignedVec::new(fft_size),
  };
  
  let chunk_size = fft_size - n_k + 1;
  let n_chunk = (n_x as f64 / chunk_size as f64).ceil() as usize;
  (0..n_chunk).for_each(|i_chunk| {
    let Store { fft_plan, ifft_plan, conv_io, fft_tmp } = &mut store;

    let in_begin = i_chunk * chunk_size;
    let in_end = ((i_chunk + 1) * chunk_size).min(n_x);
    assert!(in_end <= n_x);
    assert!(in_begin < in_end);
    let real_chunk_size = in_end - in_begin;
    conv_io[..real_chunk_size].iter_mut().zip(x[in_begin..in_end].iter()).for_each(|(a, b)| *a = Complex32::new(*b, 0.0));
    fill(&mut conv_io[real_chunk_size..], Complex32::zero());
    fft_plan.c2c(conv_io, fft_tmp).unwrap();
    fft_tmp.iter_mut().zip(fd_krnl.iter()).for_each(|(v, k)| *v *= *k);
    ifft_plan.c2c(fft_tmp, conv_io).unwrap();
    out[in_begin..in_begin + real_chunk_size + n_k - 1].iter_mut().zip(conv_io[..real_chunk_size + n_k - 1].iter()).for_each(|(x, v)| *x += *v);
  });
}