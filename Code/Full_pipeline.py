import yt
import trident
import numpy as np
import matplotlib.pyplot as plt
import caesar
import astropy
from astropy.table import Table, join, vstack
import os
import time
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, MaxNLocator, FixedLocator
from matplotlib.gridspec import GridSpec
import pygad as pg

plt.style.use('classic')
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.size'] = 8
plt.rcParams['xtick.minor.size'] = 4
plt.rcParams['ytick.major.size'] = 8
plt.rcParams['ytick.minor.size'] = 4
plt.tick_params(axis='both',direction='in')


def get_mass_luminosity_subset(mass_low, mass_high):
    ## Find all groups between mass range
    ## Ensure groups have at least 3 members > L* in SDSS r
    halo_mass_subset = [np.log10(i.masses['total']) for i in caesar_cat.halos if ((np.log10(i.masses['total']) >= mass_low) and (np.log10(i.masses['total']) <= mass_high) and (len(i.galaxy_index_list) >= 3))]
    halo_loc_subset = [i.minpotpos for i in caesar_cat.halos if ((np.log10(i.masses['total']) >= mass_low) and (np.log10(i.masses['total']) <= mass_high) and (len(i.galaxy_index_list) >= 3))]
    halo_rvir_subset = [i.virial_quantities['r200c'] for i in caesar_cat.halos if ((np.log10(i.masses['total']) >= mass_low) and (np.log10(i.masses['total']) <= mass_high) and (len(i.galaxy_index_list) >= 3))]
    halo_ids_subset = [i.GroupID for i in caesar_cat.halos if ((np.log10(i.masses['total']) >= mass_low) and (np.log10(i.masses['total']) <= mass_high) and (len(i.galaxy_index_list) >= 3))]
    
    mass_sample = np.asarray([])
    location_sample = []
    rvir_sample = []
    id_sample = np.asarray([])

    for gg,g in enumerate(halo_ids_subset):# gg is index, g value
        members = np.asarray([i.GroupID for i in caesar_cat.galaxies if (i.parent_halo_index == g)])
        mag = np.asarray([caesar_cat.galaxies[i].absmag['sdss_r'] for i in members])
        lum = 10**((mag + 20.83)/-2.5) ### Blanton et al. 2001 AJ 121 2358 and https://www.astro.umd.edu/~richard/ASTRO620/LumFunction-pp.pdf
        if (len(lum[lum >=1 ]) >= 3) and (len(lum[lum >=1 ]) <= 7):
            id_sample = np.append(id_sample,halo_ids_subset[gg])
            mass_sample = np.append(mass_sample,halo_mass_subset[gg])
            # rvir_sample = np.append(rvir_sample,halo_rvir_subset[gg])
            rvir_sample.append(halo_rvir_subset[gg])
            # location_sample = np.append(location_sample,halo_loc_subset[gg])
            location_sample.append(halo_loc_subset[gg])

    return mass_sample, location_sample, rvir_sample, id_sample

def create_ray_grid2D(center,radius,num):
    # center = center of group
    # radius = group virial radius
    # num = number of sightlines on each side
    # returns lists of start and end locations in xy and along z
    x_start = center.copy()[0] - 1.5 * radius
    x_end = center.copy()[0] + 1.5 * radius
    x_values = np.linspace(x_start,x_end,num=num)
    
    y_start = center.copy()[1] - 1.5 * radius
    y_end = center.copy()[1] + 1.5 * radius
    y_values = np.linspace(y_start,y_end,num=num)
    
    z_start = center.copy()[2] - 2.0 * radius
    z_end = center.copy()[2] + 2.0 * radius
    
    xx,yy = np.meshgrid(x_values,y_values)
    coords = np.vstack([xx.ravel(),yy.ravel()])
    ray_starts = []
    ray_ends = []
    
    for i in range(coords.shape[1]):
        ray_starts.append(snap.arr(np.asarray([coords[0][i],coords[1][i],z_start]),'kpccm'))
        ray_ends.append(snap.arr(np.asarray([coords[0][i],coords[1][i],z_end]),'kpccm'))
    return ray_starts, ray_ends

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


def fit_vp_pipeline(ray, ion, wave, vel, flux, error, z, logN_bounds=[11, 19], b_bounds=[5,200], min_region_width=5, N_sigma=1.5):#, plot_fit=True, write_params=True):
    ## ion = ion name from pygad
    ## z = Snapshot redshift
    ##
    #### Algorithm ####
    # 1) Read spectrum and trim
    # 2) Test for saturation
    # 3) Detect absorption regions
    # 4) Fit detected regions
    # 5) Return table with VP params

    wave_subset = wave[(vel >= -1500) & (vel <= 1500)]
    flux_subset = flux[(vel >= -1500) & (vel <= 1500)]
    error_subset = error[(vel >= -1500) & (vel <= 1500)]

    sat_flag = False
    if np.min(flux) <= 0.2:
        sat_flag = True

    ## Creating output table
    t = Table()

    regions, _ = pg.analysis.vpfit.find_regions(wave_subset, flux_subset, error_subset, min_region_width=min_region_width, N_sigma=N_sigma, extend=True)
    print('Found %d absorption regions for '%len(regions)+ion)

    if len(regions) > 0:
        fit, chisq = pg.analysis.vpfit.fit_profiles(ion, wave_subset, flux_subset, error_subset, chisq_lim=0.25, max_lines=6, mode="Voigt", logN_bounds=logN_bounds, b_bounds=b_bounds, min_region_width=min_region_width, N_sigma=N_sigma, extend=True)

        vels = wave_to_vel(fit['l'],float(pg.analysis.absorption_spectra.lines[pg_ion]['l'].split()[0]))
        for v in vels:
            t['Sightline'] = np.ones(len(fit['N']))*ray
            t['Species'] = [ion.replace(' ','')]*len(fit['N'])
            t['EW(mA)'] = fit['EW']*1000. # mA
            t['dEW(mA)'] = np.nan # mA
            t['N'] = fit['N']
            t['dN'] = fit['dN']
            t['b'] = fit['b']
            t['db'] = fit['db']
            t['v'] = wave_to_vel(fit['l'],float(pg.analysis.absorption_spectra.lines[pg_ion]['l'].split()[0]))
            v_up = wave_to_vel((fit['l'] + fit['dl']),float(pg.analysis.absorption_spectra.lines[pg_ion]['l'].split()[0]))
            v_low = wave_to_vel((fit['l'] - fit['dl']),float(pg.analysis.absorption_spectra.lines[pg_ion]['l'].split()[0]))
            t['dv'] = ((v_up - t['v']) + (t['v'] - v_low))/2.
            t['l'] = fit['l']
            t['dl'] = fit['dl']
            t['UpLim'] = np.ones(len(fit['N'])) * False
            t['Sat'] = np.ones(len(fit['N'])) * sat_flag
            t['Chisq'] = np.ones(len(fit['N'])) * chisq


    else: 
        ## Measure EW and rms over +- 50 km/s and determine N upper limit
        vel_subset = wave_to_vel(wave_subset,float(pg.analysis.absorption_spectra.lines[pg_ion]['l'].split()[0]))
        # spec['vel'] = vel
        wave_pm100 = wave_subset[(vel_subset >= -50.) & (vel_subset <= 50.)]
        flux_pm100 = flux_subset[(vel_subset >= -50.) & (vel_subset <= 50.)]
        error_pm100 = error_subset[(vel_subset >= -50.) & (vel_subset <= 50.)]
        ew_pm100 = pg.analysis.vpfit.EquivalentWidth(flux_pm100,wave_pm100) * 1000 # mA
        dew_pm100 = ((np.sqrt(np.sum(error_pm100**2)))*abs(wave_pm100[1] - wave_pm100[0])) * 1000 # mA

        N_lim = EW_to_N(pg_ion,dew_pm100)
        t['Sightline'] = [ray]
        t['Species'] = [ion.replace(' ','')]
        t['EW(mA)'] = ew_pm100 # mA
        t['dEW(mA)'] = dew_pm100 # mA
        t['N'] = N_lim
        t['dN'] = np.nan
        t['b'] = np.nan
        t['db'] = np.nan
        t['v'] = 0
        t['dv'] = 50
        t['l'] = np.nan
        t['dl'] = np.nan
        t['UpLim'] = np.ones(1) * True
        t['Sat'] =  np.ones(1) * sat_flag
        t['Chisq'] = np.nan 

    return sat_flag, t

# start_time = time.strftime('%c')

z = 0.1378078834409786
caesar_cat = caesar.load('/Users/tjmccab2/Dropbox/SIMBA_IGrM/Caesar_cats/m100n1024_143.hdf5')

snap_location = Table.read('/Users/tjmccab2/Dropbox/SIMBA_IGrM/Simulations/location.txt',format='ascii.no_header')
snap_path = snap_location['col1'][0]
print('\n')
snap = yt.load(snap_path[1:-1])

num_rays = 100 #10

halo_masses, halo_locs, halo_rvir, halo_ids = get_mass_luminosity_subset(12.89, 13.61) ## Just COS-IGrM range for now
print(len(halo_masses))
# exit()
for grp in range(4,len(halo_masses)):
    # if grp == 1:
    #     exit()

    print('Group %d at time: %s'%(halo_ids[grp],time.strftime("%H:%M:%S", time.localtime())))
    print(str(grp+1)+'/'+str(len(halo_masses)))
    print('\n')

    data_path = '/Users/tjmccab2/Dropbox/SIMBA_IGrM/Results/Group_%d/'%halo_ids[grp]
    try:
        os.makedirs(data_path)
        print('Creating directory %s'%data_path)
    except FileExistsError:
        pass

    mass = halo_masses[grp]
    position = halo_locs[grp]
    rvir = halo_rvir[grp]

    rays_start, rays_end = create_ray_grid2D(position,rvir,num_rays)

    t = Table()
    t['Sightline'] = np.arange(1,len(rays_start)+1)
    t['Start_x'] = [i[0].value for i in rays_start]
    t['Start_y'] = [i[1].value for i in rays_start]
    t['Start_z'] = [i[2].value for i in rays_start]
    t['End_x'] = [i[0].value for i in rays_end]
    t['End_y'] = [i[1].value for i in rays_end]
    t['End_z'] = [i[2].value for i in rays_end]
    t['Units'] = [i[0].units for i in rays_start]
    print(t)
    t.write(data_path+'Grp_%d_Ray_info.txt'%(halo_ids[grp]),format='ascii.fixed_width',overwrite=True)

    ## Create output results table
    results_table = Table()
    names = ('Sightline','Species','EW(mA)','dEW(mA)','N', 'dN', 'b', 'db', 'v', 'dv', 'l', 'dl', 'UpLim', 'Sat' , 'Chisq')
    empty = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    results_table.add_columns(empty,names=names)
    results_table['Sightline'] = results_table['Sightline'].astype(int)
    results_table['Species'] = results_table['Species'].astype(str)


    ### Generating Spectra
    line_list = ['O VI 1032']
    for i in range(len(rays_start)):
        try:
            ray = trident.make_simple_ray(snap,start_position=rays_start[i],end_position=rays_end[i],data_filename=None,lines=line_list)
        
            sg = trident.SpectrumGenerator('COS-G130M',dlambda=0.04)# dlambda doesn't work. Still 0.01A per pix.
            sg.make_spectrum(ray,lines=line_list)
            sg.apply_lsf()
            sg.add_gaussian_noise(20)

            pg_ion = 'OVI1031' # Trident to Pygad syntax

            #### Binning to COS resolution 2 pix per 20km/s (0.08A) Resolution
            #### Shift to rest frame

            wave_binned = np.arange(sg.lambda_min.value,sg.lambda_max.value,0.04)
            flux_binned = np.interp(wave_binned,sg.lambda_field.value,sg.flux_field)
            error_binned = np.interp(wave_binned, sg.lambda_field.value, sg.error_func(sg.flux_field))
            wave_binned = wave_binned/(1+z)
            vel_binned = wave_to_vel(wave_binned,float(pg.analysis.absorption_spectra.lines[pg_ion]['l'].split()[0]))


            min_region_width = 3 #pix
            N_sigma = 0.5
            logN_bounds = [12,16]
            b_bounds = [5,100]

            saturation_flag, output_table = fit_vp_pipeline(i+1,pg_ion, wave_binned,vel_binned, flux_binned,error_binned, z, logN_bounds, b_bounds, min_region_width, N_sigma)
            
            # print(output_table)
            # print(results_table)
            results_table = vstack((results_table,output_table))

        except RuntimeError:
            empty_row = Table()
            names = ('Sightline','Species','EW(mA)','dEW(mA)','N', 'dN', 'b', 'db', 'v', 'dv', 'l', 'dl', 'UpLim', 'Sat' , 'Chisq')
            row = [[i+1],[np.nan],[np.nan],[np.nan],[np.nan],[np.nan],[np.nan],[np.nan],[np.nan],[np.nan],[np.nan],[np.nan],[np.nan],[np.nan],[np.nan]]
            empty_row.add_columns(row,names=names)
            empty_row['Sightline'] = empty_row['Sightline'].astype(int)
            empty_row['Species'] = empty_row['Species'].astype(str)
            results_table = vstack((results_table,empty_row))

    results_table.write(data_path + 'Grp_%d_fitting_results.txt'%halo_ids[grp],format='ascii.fixed_width',overwrite=True)
