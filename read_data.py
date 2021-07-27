# Read data from different instruments

import numpy as np
import netCDF4 as nc4
import h5py
import hdf
import os
from csat import MODIS
import datetime
import misc_codes

# read MODIS
def read_modis(f_name_modis, modis_vars):
    modis_data = hdf.read_hdf4(f_name_modis, modis_vars)
    modis_data['Solar_Zenith_gridded'] = MODIS.field_interpolate(modis_data['Solar_Zenith'])
    modis_data['Latitude_1km'] = MODIS.field_interpolate(modis_data['Latitude'])
    modis_data['Longitude_1km'] = MODIS.field_interpolate(modis_data['Longitude'])
    modis_data['Cloud_Fraction_1km'] = MODIS.field_interpolate(modis_data['Cloud_Fraction'])
    #modis_latlon = hdf.read_hdf4(f_name_modis, ['Latitude', 'Longitude'])
    for name in modis_data.keys():
        if modis_data[name].shape[1] == 1354:
            modis_data[name] = modis_data[name][:, :-4]
    return(modis_data)

def short_modis_swath_data(modis_data, delta_l, delta_r):
    # Only data along the swath
    # Input: delta_l: number of pixels left of center
    #        delta_r: number piexls right of center
    dims = modis_data['Cloud_Effective_Radius'].shape[1]//2
    modis_data_short = {}
    for var in modis_data.keys():
        modis_data_short[var] = modis_data[var][:, dims-delta_l:dims+delta_r]
    return modis_data_short

# MODIS Ed (files from from barat that were  created by Ed)
def read_modis_ed(fname_modis_l3, modis_l3_vars):
    modis_l3_data = {}
    nc_h = nc4.Dataset(fname_modis_l3)
    for var in modis_l3_vars:
        modis_l3_data[var] = np.squeeze(nc_h[var])
    return modis_l3_data
  
# Calipso
def read_calipso_l2(fname_calipso_l2, calipso_l2_vars):
    calipso_data_l2 = hdf.read_hdf4(fname_calipso_l2, calipso_l2_vars)
    return calipso_data_l2

# CALIPSO L1
def read_calipso_l1(fname_calipso_l1, calipso_l1_vars):
    calipso_data_l1 = hdf.read_hdf4(fname_calipso_l1, calipso_l1_vars)
    return calipso_data_l1

# CloudSat
def read_cldsat(fname_cldsat,cldsat_vars):
    data_cldsat = hdf.read_hdf4(fname_cldsat, cldsat_vars, vdata=True)
    return data_cldsat

# ECMWF
def read_ecmwf(fname_ecmwf, ecmwf_vars):
    ecmwf_data = {}
    nc_h = nc4.Dataset(fname_ecmwf)
    for var in ecmwf_vars:
        ecmwf_data[var] = nc_h[var]
    return ecmwf_data

# ECMWF-ERA5
def read_ERA5(fname_ecmwf, ecmwf_vars):
    ecmwf_data = {}
    nc_h = nc4.Dataset(fname_ecmwf)
    for var in ecmwf_vars:
        ecmwf_data[var] = nc_h[var]
    return ecmwf_data

# Calculate Nd
def es(temp):
    '''Returns SVP in Pa if temp is in K'''
    A,B,C = 6.1094, 17.625, 243.04
    t=temp-273.15
    return A*np.exp((B*t)/(C+t))*100

def sat_lapse_rate(T,p):
    g = 9.81
    H = 2501000
    ep = 0.622
    Rsd = 287
    cpd = 1003.5
    #return g * (1+((H*ep*es(T))/(Rsd*(p-es(T))*T))) / (cpd + ((H**2*ep**2*es(T))/(Rsd*(p-es(T))*T**2)))
    return g * (1 + ((H * ep * es(T)) / (Rsd * (p - es(T)) * T))) / (cpd + ((H ** 2 * ep ** 2 * es(T)) / (Rsd * (p - es(T)) * T ** 2)))

def calculate_Nd_adjust(re, tau, T=None, P=None):
    # calculating Cw based on T and P
    T = T or 275
    P = P or 95000

    Qext = 2
    ro_w = 997*10**3     #[gr*m^-3]

    Cp = 1004 # [J/kg K]
    ro_a = 1.2 # air density [kg/m3]
    Lv =  2.5 * 10**6 # latent heat of vaporization [J/kg]
    gamma_d = 9.81/Cp
    f_ad = 0.8

    Cw = f_ad * ((ro_a * Cp * (gamma_d - sat_lapse_rate(T, P)) / (Lv)) * 1000)  # eq 14 in Grosvenor 2018 [gr*m^-4]
    gamma = ((5**0.5)/(2*np.pi*0.8)) * (Cw/(Qext*ro_w))**0.5

    N = (gamma * tau ** 0.5 * (re * (1e-6)) ** (-5. / 2)) * 1e-6

    '''
    A = Cw/((4/3)*np.pi*ro_w)
    a1 = (5*A)/(3*np.pi*Qext)
    N = (a1 ** (0.5) * tau ** 0.5 * (re * (1e-6)) ** (-5. / 2)) * 1e-6  # because it is m^3 to cm^3
    '''
    return(N)

def calculate_Nd_fixed(re, tau):
    # Using a fixed gamma of 1.37e-5 (Quaas)
    Qext = 2
    ro = 997*10**3     #[gr*m^3]
    A = Cw/((4/3)*np.pi*ro)
    a1 = 5*A/(3*np.pi*Qext)
    N = ((a1**(0.5)* tau**(0.5)* (re*10**(-6))**(-5/2))*10**-6);   #because it is m^3 to cm^3

    gamma = 1.37e-5  # constant in units of [m^-0.5] 1.37 (1.25e-5 if from Szczodrak et al 1994)
    modis_nd = (gamma * tau ** 0.5 * ( re * (1e-6)) ** (-5. / 2)) * 1e-6
    return modis_nd

# Read MACGIC sounding data
def read_magson(fname_magson, magson_vars):
    magson_data = {}
    nc_h = nc4.Dataset(fname_magson)
    for var in magson_vars:
        magson_data[var] = nc_h[var]
    return magson_data

# Read GOCART data
def get_gocart(f_name, vars):
    data_out = {}
    nc_h = nc4.Dataset(f_name)
    data_out['lat'] = nc_h['lat'][:]
    data_out['lon'] = nc_h['lon'][:]
    for var in vars:
        print (var)
        data_out[var]=nc_h[var][:]
    data_out['time'] = {}
    time_temp = np.zeros(nc_h['time'][:].size)
    time_temp = nc_h['time'][:]
    for i,t in enumerate(time_temp):
        data_out['time'][i] = misc_codes.get_time_since([t], 1900, 1, 1)
    return data_out
