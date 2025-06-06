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
    
    halo_loc_subset = [i.minpotpos.value for i in caesar_cat.halos if ((np.log10(i.masses['total']) >= mass_low) and (np.log10(i.masses['total']) <= mass_high) and (len(i.galaxy_index_list) >= 3))]
    halo_rvir_subset = [i.virial_quantities['r200c'] for i in caesar_cat.halos if ((np.log10(i.masses['total']) >= mass_low) and (np.log10(i.masses['total']) <= mass_high) and (len(i.galaxy_index_list) >= 3))]
    halo_ids_subset = [i.GroupID for i in caesar_cat.halos if ((np.log10(i.masses['total']) >= mass_low) and (np.log10(i.masses['total']) <= mass_high) and (len(i.galaxy_index_list) >= 3))]
    vel_disp_subset = [i.velocity_dispersions['total'].value for i in caesar_cat.halos if ((np.log10(i.masses['total']) >= mass_low) and (np.log10(i.masses['total']) <= mass_high) and (len(i.galaxy_index_list) >= 3))]
    number_subset = [len(i.galaxy_index_list) for i in caesar_cat.halos if ((np.log10(i.masses['total']) >= mass_low) and (np.log10(i.masses['total']) <= mass_high) and (len(i.galaxy_index_list) >= 3))]


    halo_m200c_subset = [i.virial_quantities['m200c'] for i in caesar_cat.halos if ((np.log10(i.masses['total']) >= mass_low) and (np.log10(i.masses['total']) <= mass_high) and (len(i.galaxy_index_list) >= 3))]

    mass_sample = np.asarray([])
    location_sample = []
    rvir_sample = []
    id_sample = np.asarray([])
    vel_disp_sample = np.asarray([])
    number_sample = np.asarray([])

    m200c_sample = np.asarray([])

    vel_disp_subset = [float(i) for i in vel_disp_subset]

    for gg,g in enumerate(halo_ids_subset):# gg is index, g value
        members = np.asarray([i.GroupID for i in caesar_cat.galaxies if (i.parent_halo_index == g)])
        mag = np.asarray([caesar_cat.galaxies[i].absmag['sdss_r'] for i in members])
        lum = 10**((mag + 20.83)/-2.5) ### Blanton et al. 2001 AJ 121 2358 and https://www.astro.umd.edu/~richard/ASTRO620/LumFunction-pp.pdf
        if (len(lum[lum >=1 ]) >= 3) and (len(lum[lum >=1 ]) <= 7):
            id_sample = np.append(id_sample,int(halo_ids_subset[gg]))
            mass_sample = np.append(mass_sample,halo_mass_subset[gg])
            # rvir_sample = np.append(rvir_sample,halo_rvir_subset[gg])
            rvir_sample.append(halo_rvir_subset[gg])
            # location_sample = np.append(location_sample,halo_loc_subset[gg])
            location_sample.append(halo_loc_subset[gg])
            vel_disp_sample = np.append(vel_disp_sample, vel_disp_subset[gg])
            number_sample = np.append(number_sample,number_subset[gg])

            m200c_sample = np.append(m200c_sample, halo_m200c_subset[gg])  # Append the m200c mass
    return mass_sample, location_sample, rvir_sample, id_sample, vel_disp_sample, number_sample, m200c_sample



caesar_cat = caesar.load('/Users/tjmccab2/Dropbox/SIMBA_IGrM/Caesar_cats/m100n1024_143.hdf5')
halo_masses, halo_locs, halo_rvir, halo_ids, vel_disp, number,m200c_sample = get_mass_luminosity_subset(12.89, 13.61) ## Just COS-IGrM range for now
print(len(halo_masses),len(vel_disp),len(number))

# print(halo_locs,'\n')
loc_x = [float(i[0]) for i in halo_locs]
loc_y = [float(i[1]) for i in halo_locs]
loc_z = [float(i[2]) for i in halo_locs]
# print(loc_x)

t = Table()
t['Grp_ID'] = halo_ids
t['Halo_mass'] = halo_masses
t['R_vir'] = halo_rvir
t['Vel_disp'] = vel_disp
t['Number'] = number
t['Pos_x'] = loc_x
t['Pos_y'] = loc_y
t['Pos_z'] = loc_z
t['M200c'] = m200c_sample

#t.write('/Users/tjmccab2/Dropbox/SIMBA_IGrM/Results/Grp_info.txt',format='ascii.fixed_width',overwrite=True)
t.write('Grp_info_with_m200c.txt',format='ascii.fixed_width',overwrite=True)


