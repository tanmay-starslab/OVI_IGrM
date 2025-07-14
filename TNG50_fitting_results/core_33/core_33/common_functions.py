import yt
import trident
import numpy as np
import matplotlib.pyplot as plt
import astropy
from astropy.table import Table, join, vstack
import os
import time
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, MaxNLocator, FixedLocator
from matplotlib.gridspec import GridSpec
import pygad as pg
from trident import LSF
import h5py

def shift_to_restframe(wave,z):
    rest = wave / (1 + z)
    return rest

def wave_to_vel(wavelength,center_wave):
    vel = 2.99792458e5 * ((wavelength/center_wave) - 1)
    return vel 

def vel_to_wave(vel,center_wave):
    wave = (1. + (vel/2.99792458e5)) * center_wave
    return wave

def match_ion(line):
    if line == 'H I 1216':
        ion = 'H1215'
        return ion
    elif line == 'H I 1026':
        ion = 'H1025'
        return ion
    elif line == 'Ly c':
        ion = 'H972'
        return ion
    elif line == 'O VI 1032':
        ion = 'OVI1031'
        return ion
    elif line == 'O VI 1038':
        ion = 'OVI1037'
        return ion
    elif line == 'Si II 1193':
        ion = 'SiII1193'
        return ion
    elif line == 'Si II 1190':
        ion = 'SiII1190'
        return ion
    elif line == 'Si III 1206':
        ion = 'SiIII1206'
        return ion
    elif line == 'N V 1239':
        ion = 'NV1238'
        return ion
    elif line == 'C II 1036':
        ion = 'CII1036'
        return ion
    else:
        print('Error: Check to make sure Trident ion matches Pygad')

def test_for_saturation(strong_line, flux, z):
    if strong_line == 'H I 1216':
        if np.min(flux) <= 0.05:
            use_line = 'H I 1026'
            saturated = True
        else:
            use_line = strong_line
            saturated = False
    elif strong_line == 'H I 1026':
        if np.min(flux) <= 0.25:
            use_line = 'H I 972'
            saturated = True
        else:
            use_line = strong_line
            saturated = False
    elif strong_line == 'O VI 1032':
        if np.min(flux) <= 0.25:
            use_line = 'O VI 1038'
            saturated = True
        else:
            use_line = strong_line
            saturated = False
    elif strong_line == 'Si II 1193':
        if np.min(flux) <= 0.25:
            use_line = 'Si II 1190'
            saturated = True
        else:
            use_line = strong_line
            saturated = False
    else:
        use_line = strong_line
        saturated = False
    return saturated, use_line

def EW_to_N(pg_ion,ew_err):

    ## Following Draine eq. 9.15

    f = float(pg.analysis.absorption_spectra.lines[pg_ion]['f'])
    l = float(pg.analysis.absorption_spectra.lines[pg_ion]['l'].split()[0])
    N = 1.13e12 * (3 * ew_err * 1.0e-11) / f / (l * 1.0e-8)**2

    return np.log10(N)

def N_to_EW(pg_ion,logN):
    
    ## Following Draine eq. 9.15

    f = float(pg.analysis.absorption_spectra.lines[pg_ion]['f'])
    l = float(pg.analysis.absorption_spectra.lines[pg_ion]['l'].split()[0])
    EW = f * (l * 1.0e-8)**2 * 10**(logN) / 1.13e12

    return EW

def fit_vp_pipeline_new(ray, ion, wave, vel, flux, error, z, logN_bounds=[11, 19], b_bounds=[5, 200], min_region_width=5, N_sigma=1.5,chisq_lim=0.25):
    """
    Fits Voigt profiles to absorption regions and generates a table of results.

    Parameters:
    - ray: Sightline ID
    - ion: Ion name
    - wave: Wavelength array
    - vel: Velocity array
    - flux: Flux array
    - error: Error array
    - z: Redshift
    - logN_bounds: Logarithmic column density bounds for the fit
    - b_bounds: Doppler parameter bounds for the fit
    - min_region_width: Minimum width of the region to consider for fitting (in pixels)
    - N_sigma: Detection threshold in sigma for the region

    Returns:
    - sat_flag: Boolean indicating if the line is saturated
    - t: Astropy Table containing fitting results
    """
    pg_ion = 'OVI1031'
    wave_subset = wave[(vel >= -2500) & (vel <= 2500)]
    flux_subset = flux[(vel >= -2500) & (vel <= 2500)]
    error_subset = error[(vel >= -2500) & (vel <= 2500)]
    #wave_subset = wave[(vel >= -800) & (vel <= 800)]
    #flux_subset = flux[(vel >= -800) & (vel <= 800)]
    #error_subset = error[(vel >= -800) & (vel <= 800)]

    sat_flag = False
    if np.min(flux) <= 0.2:
        sat_flag = True

    # Creating output table
    t = Table(names=('Sightline', 'Species', 'EW(mA)', 'dEW(mA)', 'N', 'dN', 'b', 'db', 'v', 'dv', 'l', 'dl', 'UpLim', 'Sat', 'Chisq'),
              dtype=['i4', 'S10', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'bool', 'bool', 'f8'])

    # Detect absorption regions
    regions, _ = pg.analysis.vpfit.find_regions(
        wave_subset, flux_subset, error_subset, min_region_width=min_region_width, N_sigma=N_sigma, extend=True)
    print(f'Found {len(regions)} absorption regions for {ion}')

    if len(regions) > 0:
        # Fit detected regions
        fit = pg.analysis.vpfit.fit_profiles(
            ion, wave_subset, flux_subset, error_subset,
            chisq_lim=chisq_lim, max_lines=4, mode="Voigt",
            logN_bounds=logN_bounds, b_bounds=b_bounds,
            min_region_width=min_region_width, N_sigma=N_sigma, extend=True)
        
	    
        print(f"Type of fit: {type(fit)}")
        print(f"Fit keys: {list(fit.keys()) if isinstance(fit, dict) else 'Not a dict'}")
        print(f"Fit content: {fit}")
        print(f"[DEBUG] Attempting to access fit['chisq']")
        chisq = fit['chisq']
        print(f"[DEBUG] chisq = {chisq}")
        print(f"[DEBUG] chisq type: {type(chisq)}")
        if isinstance(chisq, np.ndarray):
            print(f"[DEBUG] chisq shape: {chisq.shape}")
        vels = wave_to_vel(fit['l'], float(pg.analysis.absorption_spectra.lines[pg_ion]['l'].split()[0]))

        # for v in vels:
        #     for j in range(len(fit['EW'])):  # Loop over components
        #         t.add_row((
        #             ray,
        #             ion.replace(' ', ''),
        #             fit['EW'][j] * 1000,  # EW in mA for the j-th component
        #             np.nan,  # Placeholder for dEW
        #             fit['N'][j],
        #             fit['dN'][j],
        #             fit['b'][j],
        #             fit['db'][j],
        #             wave_to_vel(fit['l'][j], float(pg.analysis.absorption_spectra.lines[pg_ion]['l'].split()[0])),
        #             (wave_to_vel(fit['l'][j] + fit['dl'][j], float(pg.analysis.absorption_spectra.lines[pg_ion]['l'].split()[0]))
        #             - wave_to_vel(fit['l'][j], float(pg.analysis.absorption_spectra.lines[pg_ion]['l'].split()[0]))) / 2.0,
        #             fit['l'][j],
        #             fit['dl'][j],
        #             False,  # Not an upper limit
        #             sat_flag,
        #             fit['chisq'][j]  # Assign chisq per component
        #         ))
        for j in range(len(fit['EW'])):  # Loop over components
            t.add_row((
                ray,
                ion.replace(' ', ''),
                fit['EW'][j] * 1000,  # EW in mA for the j-th component
                np.nan,  # Placeholder for dEW
                fit['N'][j],
                fit['dN'][j],
                fit['b'][j],
                fit['db'][j],
                wave_to_vel(fit['l'][j], float(pg.analysis.absorption_spectra.lines[pg_ion]['l'].split()[0])),
                (wave_to_vel(fit['l'][j] + fit['dl'][j], float(pg.analysis.absorption_spectra.lines[pg_ion]['l'].split()[0]))
                - wave_to_vel(fit['l'][j], float(pg.analysis.absorption_spectra.lines[pg_ion]['l'].split()[0]))) / 2.0,
                fit['l'][j],
                fit['dl'][j],
                False,  # Not an upper limit
                sat_flag,
                fit['chisq'][j]  # Assign chisq per component
            ))

    else:
        fit = None
        # No detected regions: Calculate upper limit
        vel_subset = wave_to_vel(wave_subset, float(pg.analysis.absorption_spectra.lines[pg_ion]['l'].split()[0]))
        wave_pm100 = wave_subset[(vel_subset >= -50.0) & (vel_subset <= 50.0)]
        flux_pm100 = flux_subset[(vel_subset >= -50.0) & (vel_subset <= 50.0)]
        error_pm100 = error_subset[(vel_subset >= -50.0) & (vel_subset <= 50.0)]
        ew_pm100 = pg.analysis.vpfit.EquivalentWidth(flux_pm100, wave_pm100) * 1000  # mA
        print(f'EW: {ew_pm100}')
        print(f'error_pm100: {error_pm100}')
        print(f'abs(wave_pm100[1] - wave_pm100[0])= {abs(wave_pm100[1] - wave_pm100[0])}')
        dew_pm100 = ((np.sqrt(np.sum(error_pm100**2))) * abs(wave_pm100[1] - wave_pm100[0])) * 1000  # mA
        print(f'dEW: {dew_pm100}')
        N_lim = EW_to_N(pg_ion, dew_pm100)

        # Add a single row for the non-detection case
        t.add_row((
            ray,
            ion.replace(' ', ''),
            ew_pm100,  # EW in mA
            dew_pm100,  # dEW in mA
            N_lim,
            np.nan,  # dN placeholder
            np.nan,  # b placeholder
            np.nan,  # db placeholder
            0,  # velocity placeholder
            50,  # dv placeholder
            np.nan,  # wavelength placeholder
            np.nan,  # dl placeholder
            True,  # Upper limit
            sat_flag,
            np.nan  # chisq placeholder
        ))

    return sat_flag, t,fit,regions

results_table = Table(
    names=(
        'Sightline', 'Species', 'EW(mA)', 'dEW(mA)', 'N', 'dN', 'b', 'db', 'v', 
        'dv', 'l', 'dl', 'UpLim', 'Sat', 'Chisq'
    ),
    dtype=['i4', 'S10', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'bool', 'bool', 'f8']
)

line_list = ['O VI 1032']


def generate_params(fitting_data):
    """
    Generate the parameter array for Voigt profile fitting from the fitting_data dictionary.

    Parameters:
        fitting_data (dict): Dictionary containing 'N', 'b', and 'l' arrays.

    Returns:
        np.ndarray: Flattened parameter array in the required format.
    """
    # Ensure 'N', 'b', and 'l' keys are present
    if not all(key in fitting_data for key in ['N', 'b', 'l']):
        raise ValueError("fitting_data must contain 'N', 'b', and 'l' keys.")
    
    # Interleave 'N', 'b', and 'l' to create the params array
    params = np.empty((3 * len(fitting_data['N'])))
    params[0::3] = fitting_data['N']  # Column densities
    params[1::3] = fitting_data['b']  # Doppler parameters
    params[2::3] = fitting_data['l']  # Centroid wavelengths
    
    return params


print("ALL OK!")

