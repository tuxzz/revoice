use fftw::types::Flag;
use fftw::plan::{R2CPlan, C2RPlan, R2CPlan32, C2RPlan32};
use num_complex::*;
use rayon::prelude::*;

const window_coeff: [f32;4] = [0.338946, 0.481973, 0.161054, 0.018027];

