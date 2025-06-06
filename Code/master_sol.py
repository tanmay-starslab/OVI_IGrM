import os

# Template for the core processing script
SCRIPT_TEMPLATE = """#!/usr/bin/env python
import h5py
import os
import numpy as np
from astropy.convolution import convolve
from astropy.table import Table, vstack
from common_functions import fit_vp_pipeline_new, generate_params

# Global seed for reproducibility
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)

# Parameters
INPUT_FILE = "{output_dir}/core_{core_id}/core_{core_id}.hdf5"
OUTPUT_FILE = "{output_dir}/core_{core_id}/result_core_{core_id}.txt"
INTERMEDIATE_SAVE_FREQ = 50
REST_WAVELENGTH = 1031.912
Z = 0.09940180263022191
SNR_PER_RES = 10
RESOLUTION_PIX = 6
BIN_PIXELS = 3

def make_noise_array(flux, SNR, constant=True):
    if constant:
        noise = 1 / SNR * np.ones_like(flux)
    else:
        noise_std = np.mean(flux) / SNR
        noise = np.random.normal(0, noise_std, len(flux))
    return noise

def apply_lsf_to_spectrum(flux, lsf_file):
    lsf_file_path = os.path.join(os.path.dirname(__file__), lsf_file)
    lsf_kernel = np.loadtxt(lsf_file_path, usecols=1)
    flux_lsf = convolve(flux, lsf_kernel)
    np.clip(flux_lsf, 0, np.inf, out=flux_lsf)
    return flux_lsf

def bin_data(wavelength, flux, bin_pixels):
    num_bins = len(wavelength) // bin_pixels
    reshaped_flux = flux[:num_bins * bin_pixels].reshape(num_bins, bin_pixels)
    reshaped_wavelength = wavelength[:num_bins * bin_pixels].reshape(num_bins, bin_pixels)
    binned_flux = np.mean(reshaped_flux, axis=1)
    binned_wavelength = np.mean(reshaped_wavelength, axis=1)
    return binned_wavelength, binned_flux

def add_gaussian_noise(flux, SNR):
    noise_std = np.mean(flux) / SNR
    noise = np.random.normal(0, noise_std, len(flux))
    flux_noisy = flux + noise
    return flux_noisy

def process_spectra(input_file, output_file, lsf_file):
    with h5py.File(input_file, "r") as f:
        wave = f["wave"][:]
        tau_OVI = f["tau_OVI_1031"][:]
        flux = np.exp(-tau_OVI)

    results_table = Table(
        names=('Sightline', 'Species', 'EW(mA)', 'dEW(mA)', 'N', 'dN', 'b', 'db', 'v', 
               'dv', 'l', 'dl', 'UpLim', 'Sat', 'Chisq'),
        dtype=['i4', 'S10', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'bool', 'bool', 'f8']
    )

    SNR_PER_BIN = SNR_PER_RES / np.sqrt(RESOLUTION_PIX / BIN_PIXELS)

    for i, single_flux in enumerate(flux):
        try:
            flux_lsf = apply_lsf_to_spectrum(single_flux, lsf_file)
            binned_wavelength, binned_flux = bin_data(wave, flux_lsf, BIN_PIXELS)
            noise = make_noise_array(binned_flux, SNR_PER_BIN, constant=True)
            flux_noisy = add_gaussian_noise(binned_flux, SNR_PER_BIN)
            
            wave_rest = binned_wavelength / (1 + Z)
            velocity = (wave_rest - REST_WAVELENGTH) / REST_WAVELENGTH * 299792.458
            pg_ion = 'OVI1031'

            saturation_flag, output_table, fit, regions = fit_vp_pipeline_new(
                i, pg_ion, wave_rest, velocity, flux_noisy, noise, Z, [13.5, 18], [6, 100], 3, 3, 1
            )

            results_table = vstack((results_table, output_table))
        except RuntimeError:
            empty_row = Table(
                [[i], [np.nan], [np.nan], [np.nan], [np.nan], [np.nan], [np.nan], [np.nan], [np.nan], [np.nan], 
                 [np.nan], [np.nan], [False], [False], [np.nan]],
                names=results_table.colnames
            )
            results_table = vstack((results_table, empty_row))

        if (i + 1) % INTERMEDIATE_SAVE_FREQ == 0:
            results_table.write(output_file, format="ascii.fixed_width", overwrite=True)

    results_table.write(output_file, format="ascii.fixed_width", overwrite=True)

if __name__ == "__main__":
    process_spectra(INPUT_FILE, OUTPUT_FILE, lsf_file="COS_G130M_1150.txt")
"""

# Number of groups and cores
BASE_DIR = "/scratch/tsingh65/TNG50-1_091_AbsSpectra/results"
NUM_GROUPS = 14
N_CORES = 100

for group_id in range(NUM_GROUPS):
    group_dir = os.path.join(BASE_DIR, f"group_{group_id}")
    os.makedirs(group_dir, exist_ok=True)

    for core_id in range(N_CORES):
        script_dir = os.path.join(group_dir, f"core_{core_id}")
        os.makedirs(script_dir, exist_ok=True)

        script_file = os.path.join(script_dir, f"process_core_{core_id}.py")
        print(f"Creating process_core_{core_id}.py in {script_dir}...")

        with open(script_file, "w") as f:
            f.write(SCRIPT_TEMPLATE.format(
                core_id=core_id,
                output_dir=group_dir
            ))
