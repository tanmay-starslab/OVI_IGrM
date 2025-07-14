#!/usr/bin/env python
import h5py
import os
import numpy as np
from astropy.table import Table, vstack
from common_functions import fit_vp_pipeline_new, generate_params

# Global seed for reproducibility
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)

# Parameters
INPUT_FILE = "/scratch/tsingh65/TNG50-1_091_AbsSpectra_2500/results/group_12/core_33/core_33.hdf5"
OUTPUT_FILE = "/scratch/tsingh65/TNG50-1_091_AbsSpectra_2500/results/group_12/core_33/result_core_33.txt"
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

def process_spectra(input_file, output_file):
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
            binned_wavelength, binned_flux = bin_data(wave, single_flux, BIN_PIXELS)
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
    process_spectra(INPUT_FILE, OUTPUT_FILE)
