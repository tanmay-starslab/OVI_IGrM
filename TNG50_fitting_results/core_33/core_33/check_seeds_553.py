#!/usr/bin/env python

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, vstack
from common_functions import fit_vp_pipeline_new

INPUT_FILE = "core_33.hdf5"
HDF5_ALLRUNS = "result_core_33_all_noise_realizations.hdf5"
OUT_DIR = "all_outputs"
SUMMARY_FILE = os.path.join(OUT_DIR, "result_core_33_summary.txt")

REST_WAVELENGTH = 1031.912
Z = 0.09940180263022191
SNR_PER_RES = 10
RESOLUTION_PIX = 6
BIN_PIXELS = 3
VEL_WINDOW = 1000.0

os.makedirs(OUT_DIR, exist_ok=True)

def make_noise_array(flux, SNR):
    sigma = np.mean(flux) / SNR
    return np.full_like(flux, sigma)

def bin_data(wave, flux, n_pix=3):
    nb = len(wave) // n_pix
    wbin = wave[:nb * n_pix].reshape(nb, n_pix).mean(1)
    fbin = flux[:nb * n_pix].reshape(nb, n_pix).mean(1)
    return wbin, fbin

def process_seed(seed: int, h5out: h5py.File, w_rest_full: np.ndarray) -> int:
    np.random.seed(seed)

    with h5py.File(INPUT_FILE, "r") as f:
        wave = f["wave"][:]
        tau = f["tau_OVI_1031"][553:554]
        flux = np.exp(-tau)

    results = Table(
        names=('Sightline', 'Species', 'EW(mA)', 'dEW(mA)', 'N', 'dN',
               'b', 'db', 'v', 'dv', 'l', 'dl', 'UpLim', 'Sat', 'Chisq'),
        dtype=['i4', 'S10', 'f8', 'f8', 'f8', 'f8',
               'f8', 'f8', 'f8', 'f8', 'f8', 'f8',
               'bool', 'bool', 'f8']
    )

    for i, spec in enumerate(flux):
        wbin, fbin = bin_data(wave, spec, BIN_PIXELS)
        SNR_BIN = SNR_PER_RES / np.sqrt(RESOLUTION_PIX / BIN_PIXELS)
        sigma = make_noise_array(fbin, SNR_BIN)
        f_noisy = fbin + np.random.normal(0.0, sigma, len(fbin))

        w_rest = wbin / (1 + Z)
        vel = (w_rest - REST_WAVELENGTH) / REST_WAVELENGTH * 299792.458

        try:
            sat_flag, out_tbl, *_ = fit_vp_pipeline_new(
                i, "OVI1031", w_rest, vel,
                f_noisy, sigma,
                Z, [13.5, 18], [6, 100], 3, 3, 1
            )
            results = vstack((results, out_tbl))
        except RuntimeError:
            blank = Table(
                [[i], [np.nan], [np.nan], [np.nan], [np.nan], [np.nan],
                 [np.nan], [np.nan], [np.nan], [np.nan], [np.nan], [np.nan],
                 [False], [False], [np.nan]],
                names=results.colnames
            )
            results = vstack((results, blank))

        mask = (vel > -VEL_WINDOW) & (vel < VEL_WINDOW)
        png_file = os.path.join(OUT_DIR, f"result_core_33_seed{seed:03d}_spectrum.png")

        plt.figure(figsize=(8, 4))
        plt.errorbar(vel[mask], f_noisy[mask], yerr=sigma[mask],
                     fmt='o', ms=3, ecolor='lightgray', mec='k', mfc='none',
                     lw=0.8, alpha=0.9)
        plt.step(vel[mask], f_noisy[mask], where='mid',
                 color='k', lw=1.2, alpha=0.9)
        plt.axhline(1.0, color='C1', lw=1.0)
        plt.xlabel(r"Velocity  [km\,s$^{-1}$]")
        plt.ylabel("Normalized Flux")
        plt.title(f"Seed {seed:03d}   Δv = ±{VEL_WINDOW:.0f} km s$^{{-1}}$")
        plt.xlim(-VEL_WINDOW, VEL_WINDOW)
        plt.ylim(-0.1, 1.4)
        plt.tight_layout()
        plt.savefig(png_file, dpi=300)
        plt.close()

        g = h5out.create_group(f"seed_{seed:03d}")
        g.create_dataset("flux", data=f_noisy, compression="gzip")
        g.create_dataset("flux_err", data=sigma, compression="gzip")

    txt_file = os.path.join(OUT_DIR, f"result_core_33_seed{seed:03d}.txt")
    results.write(txt_file, format="ascii.fixed_width", overwrite=True)
    return len(results)

if __name__ == "__main__":
    # Pre-compute rest-frame wavelength array
    with h5py.File(INPUT_FILE, "r") as f0:
        wave0 = f0["wave"][:]
    w_rest_full, _ = bin_data(wave0, wave0, BIN_PIXELS)
    w_rest_full = w_rest_full / (1 + Z)

    summary = []

    # If starting fresh, delete old HDF5
    if os.path.exists(HDF5_ALLRUNS):
        os.remove(HDF5_ALLRUNS)

    # Create HDF5 and initialize wave_rest
    h5out = h5py.File(HDF5_ALLRUNS, "w")
    h5out.create_dataset("wave_rest", data=w_rest_full)

    for seed in range(1000):
        nrows = process_seed(seed, h5out, w_rest_full)
        summary.append((seed, nrows))
        print(f"[{seed:04d}] Rows written: {nrows}")

        # Flush and close every 100 steps
        if (seed + 1) % 100 == 0:
            h5out.close()
            h5out = h5py.File(HDF5_ALLRUNS, "a")  # reopen for appending

    h5out.close()
    np.savetxt(SUMMARY_FILE, summary, fmt="%3d %d",
               header="seed  N_rows_written")
    print(f"\nSummary written to {SUMMARY_FILE}")

