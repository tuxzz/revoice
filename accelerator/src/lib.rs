extern crate numpy;
extern crate fftw;
extern crate pyo3;
extern crate num_complex;
extern crate num_traits;
extern crate rayon;
extern crate rand;
extern crate fastapprox;

use numpy::*;
use pyo3::prelude::{pymodinit, PyModule, PyResult, Python};
use num_complex::*;

mod common;
mod mfi_envelope;
mod pyin;
mod refine_f0_stft;
//mod gvm;
mod train;

#[pymodinit]
fn accelerator(_py: Python, m: &PyModule) -> PyResult<()> {
	#[pyfn(m, "mfiCore")]
	fn mfi_core_py(_py: Python, x: &PyArray1<f32>, f0_list: &PyArray1<f32>, kernel: &PyArray1<f32>, trans: &PyArray1<f32>, fixed_f0: f32, hop_size: f64, sr: f32, win_len_fac: f32, fft_size: usize, out: &PyArray2<f32>) -> PyResult<()> {
		mfi_envelope::mfi_core(
			x.as_slice(),
			f0_list.as_slice(),
			kernel.as_slice(),
			trans.as_slice(),
			fixed_f0,
			hop_size,
			sr,
			win_len_fac,
			fft_size,
			out.as_slice_mut(),
		);
		Ok(())
	}

	#[pyfn(m, "pyinCore")]
	fn pyin_core_py(_py: Python, x: &PyArray1<f32>, pdf: &PyArray1<f32>, bias: f32, prob_threshold: f32, weight_prior: f32, hop_size: f64, valley_threshold: f32, valley_step: f32, min_freq: f32, max_freq: f32, min_win_size: f32, win_len_fac: f32, sr: f32, max_iter: usize, remove_dc: bool, out: &PyArray3<f32>) -> PyResult<()> {
		assert_eq!(out.shape()[2], 2, "output shape must be (n_hop, max_freq_prob_count, 2)");
		pyin::pyin_core(
			x.as_slice(),
			pdf.as_slice(),
			bias,
			prob_threshold,
			weight_prior,
			hop_size,
			valley_threshold,
			valley_step,
			min_freq,
			max_freq,
			min_win_size,
			win_len_fac,
			sr,
			max_iter,
			out.shape()[0],
			out.shape()[1],
			remove_dc,
			out.as_slice_mut(),
		);
		Ok(())
	}

	#[pyfn(m, "refineF0STFTCore")]
	fn refine_f0_stft_core(x: &PyArray1<f32>, in_out_f0_list: &PyArray1<f32>, hop_size: f64, fft_size: usize, win_b: f32, sr: f32, peak_search_range: f32) -> PyResult<()> {
		refine_f0_stft::refine_f0_stft_core(x.as_slice(), in_out_f0_list.as_slice_mut(), hop_size, fft_size, win_b, sr, peak_search_range);
		Ok(())
	}

	#[pyfn(m, "refreshVUVTrain")]
	fn refresh_vuv_train() -> PyResult<()> {
		train::refresh_vuv_train();
		Ok(())
	}

	#[pyfn(m, "genVUVTrainSample")]
	fn gen_vuv_train_sample(sinusoid: &PyArray1<f32>, noise: &PyArray1<f32>, freq_list: &PyArray1<f32>, window: &PyArray1<f32>, mean: &PyArray1<f32>, stdev: &PyArray1<f32>, hop_size: f64, out_magn_list: &PyArray2<f32>, fft_size: usize, sr: f32) -> PyResult<usize> {
		Ok(train::gen_vuv_train_sample(
			sinusoid.as_slice(),
			noise.as_slice(),
			freq_list.as_slice(),
			window.as_slice(),
			mean.as_slice(),
			stdev.as_slice(),
			hop_size,
			out_magn_list.as_slice_mut(),
			fft_size,
			sr,
		))
	}

	#[pyfn(m, "calcSpectrum")]
	fn calc_fft(x: &PyArray1<f32>, freq_list: &PyArray1<f32>, window: &PyArray1<f32>, hop_size: f64, out_list: &PyArray2<f32>, fft_size: usize, sr: f32) -> PyResult<()> {
		train::calc_spectrum(
			x.as_slice(),
			freq_list.as_slice(),
			window.as_slice(),
			hop_size,
			out_list.as_slice_mut(),
			fft_size,
			sr,
		);
		Ok(())
	}

	#[pyfn(m, "segconv")]
	fn segconv(x: &PyArray1<f32>, k: &PyArray1<f32>, out: &PyArray1<f32>) -> PyResult<()> {
		common::segconv(
			x.as_slice(),
			k.as_slice(),
			out.as_slice_mut(),
		);
		Ok(())
	}

	#[pyfn(m, "segconvCplx")]
	fn segconv_cplx(x: &PyArray1<Complex32>, k: &PyArray1<Complex32>, out: &PyArray1<Complex32>) -> PyResult<()> {
		common::segconv_cplx(
			x.as_slice(),
			k.as_slice(),
			out.as_slice_mut(),
		);
		Ok(())
	}

	#[pyfn(m, "segconvRealCplx")]
	fn segconv_real_cplx(x: &PyArray1<f32>, k: &PyArray1<Complex32>, out: &PyArray1<Complex32>) -> PyResult<()> {
		common::segconv_real_cplx(
			x.as_slice(),
			k.as_slice(),
			out.as_slice_mut(),
		);
		Ok(())
	}

	#[pyfn(m, "segconvSt")]
	fn segconv_st(x: &PyArray1<f32>, k: &PyArray1<f32>, out: &PyArray1<f32>) -> PyResult<()> {
		common::segconv_st(
			x.as_slice(),
			k.as_slice(),
			out.as_slice_mut(),
		);
		Ok(())
	}

	#[pyfn(m, "segconvCplxSt")]
	fn segconv_cplx_st(x: &PyArray1<Complex32>, k: &PyArray1<Complex32>, out: &PyArray1<Complex32>) -> PyResult<()> {
		common::segconv_cplx_st(
			x.as_slice(),
			k.as_slice(),
			out.as_slice_mut(),
		);
		Ok(())
	}

	#[pyfn(m, "segconvRealCplxSt")]
	fn segconv_real_cplx_st(x: &PyArray1<f32>, k: &PyArray1<Complex32>, out: &PyArray1<Complex32>) -> PyResult<()> {
		common::segconv_real_cplx_st(
			x.as_slice(),
			k.as_slice(),
			out.as_slice_mut(),
		);
		Ok(())
	}

	Ok(())
}