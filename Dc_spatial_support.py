import numpy as np
import copy
import glob
import time
import read_data

bins = {}
bins['lat'] = np.arange(-90, 91, 1)
bins['lon'] = np.arange(-180, 181, 1)

def grid_data(data, f_name_modis, ERA5_dir, output_array, factor):

    # ERA5
    MODIS_time_raw = f_name_modis.split('/')[-1].split('.')[1][1:]
    MODIS_calander_time = misc_codes.doy_to_date(int(MODIS_time_raw[0:4]), int(MODIS_time_raw[4:7]))
    MODIS_time = datetime.datetime(MODIS_calander_time[0], MODIS_calander_time[1], MODIS_calander_time[2])
    ERA5_file = 'ERA5_' + str(MODIS_time.year) + str(MODIS_time.month).zfill(2) +'.nc'
    ERA5_time_array = read_data.read_ERA5(ERA5_dir+ERA5_file, ['time'])
    for i,t in enumerate(ERA5_time_array['time'][:]):
        ERA5_time = misc_codes.get_date_time_since(1900, t / 24)
        if ERA5_time.year ==  MODIS_time.year and ERA5_time.month ==  MODIS_time.month and ERA5_time.day ==  MODIS_time.day:
            break
    print(ERA5_time, MODIS_time)
    if ERA5_time != MODIS_time:
        print('WRONG MATCH ERA5 AND MODIS')

    ERA5_data_latlon = read_data.read_ERA5(ERA5_dir+ERA5_file, ['latitude', 'longitude'])
    ERA5_data_array = read_data.read_ERA5(ERA5_dir+ERA5_file, ['t'])['t'][i,:,:,:]
    # regrid data into 1x1 degree
    x = ERA5_data_latlon['longitude'][:]-180
    y = ERA5_data_latlon['latitude'][:]
    xv, yv = np.meshgrid(x, y)
    ERA5_t1000_regrid = np.histogram2d(np.ravel(xv), np.ravel(yv), \
                               bins=[bins['lon'], bins['lat']], weights=np.ravel(ERA5_data_array[1]))[0]
    ERA5_t700_regrid = np.histogram2d(np.ravel(xv), np.ravel(yv), \
                               bins=[bins['lon'], bins['lat']], weights=np.ravel(ERA5_data_array[0]))[0]
    Num_temp_all_ERA5 = np.histogram2d(np.ravel(xv), np.ravel(yv), \
                                  bins=[bins['lon'], bins['lat']])[0].astype('int')
    temperature_1000 = (ERA5_t1000_regrid/Num_temp_all_ERA5)
    temperature_700 = (ERA5_t700_regrid/Num_temp_all_ERA5)
    LTS = misc_codes.get_LTS(temperature_1000, temperature_700)
    LTS = np.concatenate([LTS[180:,:],LTS[0:180,:]])

    # MODIS data
    data['CDNC'] = read_data.calculate_Nd_adjust(data['Cloud_Effective_Radius'], data['Cloud_Optical_Thickness']) # See the code at the end of this file
    data['CDNC'][data['Cloud_Optical_Thickness'] < 3] = np.nan
    data['CDNC'][data['Cloud_Effective_Radius'] < 3] = np.nan
    data['CDNC'][data['CDNC'] > 300] = 300

    cld_mask_copy = np.zeros(np.shape(data['Cloud_Mask_1km'][:, :, 0]))
    cld_mask_copy[data['Cloud_Mask_1km'][:,:,0]==57] = 1
    cld_mask_copy[data['Cloud_Mask_1km'][:, :, 0] == 41] = 1

    mask = [  # (data['Cloud_Phase_Optical_Properties'] == 2) *
        (data['cloud_top_temperature_1km'] > 270) *
        (data['Cloud_Multi_Layer_Flag'] < 2) *
        #(np.isfinite(data['CDNC'] + data['Cloud_Water_Path'])) *
        #(data['CDNC'] < 200)*
        #(data['Cloud_Water_Path'] > 20) *
        (data['Cloud_Optical_Thickness'] > 3)*
        (data['Cloud_Effective_Radius'] > 3)*
        (cld_mask_copy == 1)][0]

    #mask_tau = [~np.isnan(data['Cloud_Optical_Thickness'])][0] # where there is a retrieval of tau

    # Calculate Dc - critical depth for precipitation initiation
    delta_Dc = np.zeros(np.shape(data['Cloud_Effective_Radius'][mask]))  # *np.nan
    a = data['CDNC'][mask]
    b = data['Cloud_Water_Path'][mask]
    Ct = 0.0192*275 - 4.293
    for i in range(data['CDNC'][mask].shape[0]):
        #delta_Dc[i] = LUT['LWP'][int(a[i] * factor)] - b[i] # Using look up tables; see LUT codes where accuracy can be determine and LUTs can be easily re-produced.
        delta_Dc[i] = (0.033/Ct) * (int(a[i] * factor))**2 - b[i] #  Approximation - equations are derived along with Ed help; see a seperated pdf file in the codes directory.

    # Gridding - Using jpint histograms approach to make things faster
    Num_temp_all = np.histogram2d(np.ravel(data['Longitude_1km']), np.ravel(data['Latitude_1km']), \
                                  bins=[bins['lon'], bins['lat']])[0].astype('int') # All pixels within the 100x100 region

    Num_temp = np.histogram2d(data['Longitude_1km'][mask], data['Latitude_1km'][mask], \
                              bins=[bins['lon'], bins['lat']])[0].astype('int')     # Only masked pixels

    CF_tau_temp = np.histogram2d(np.ravel(data['Longitude_1km'][mask]), np.ravel(data['Latitude_1km'][mask]), \
                                   bins=[bins['lon'], bins['lat']], weights=np.ravel(data['Cloud_Fraction_1km'][mask]))[0] #CF where there is tau

    CF_MODIS_temp = np.histogram2d(np.ravel(data['Longitude_1km']), np.ravel(data['Latitude_1km']), \
                                   bins=[bins['lon'], bins['lat']], weights=np.ravel(data['Cloud_Fraction_1km']))[0] # MODIS standard CF everywhere

    CDNC_temp = np.histogram2d(data['Longitude_1km'][mask], data['Latitude_1km'][mask], \
                               bins=[bins['lon'], bins['lat']], weights=data['CDNC'][mask])[0]

    LWP_temp = np.histogram2d(data['Longitude_1km'][mask], data['Latitude_1km'][mask], \
                              bins=[bins['lon'], bins['lat']], weights=data['Cloud_Water_Path'][mask])[0]

    LWP_squared_temp = np.histogram2d(data['Longitude_1km'][mask], data['Latitude_1km'][mask], \
                              bins=[bins['lon'], bins['lat']], weights=(data['Cloud_Water_Path'][mask])**2)[0] # For std calculation

    reff_temp = np.histogram2d(data['Longitude_1km'][mask], data['Latitude_1km'][mask], \
                               bins=[bins['lon'], bins['lat']], weights=data['Cloud_Effective_Radius'][mask])[0]

    tau_temp = np.histogram2d(data['Longitude_1km'][mask], data['Latitude_1km'][mask], \
                               bins=[bins['lon'], bins['lat']], weights=data['Cloud_Optical_Thickness'][mask])[0]

    delta_Dc_temp = np.histogram2d(data['Longitude_1km'][mask], data['Latitude_1km'][mask], \
                                   bins=[bins['lon'], bins['lat']], weights=delta_Dc)[0]

    # Assuming reflectance of ocean surface = 0.06 (reference, e.g. Goren et al., 2014)
    reflectance_ocean = copy.copy(data['Atm_Corr_Refl'][:, :, 1])
    reflectance_ocean[np.isnan(data['Atm_Corr_Refl'][:, :, 1])] = 0.06
    reflectance_temp = np.histogram2d(np.ravel(data['Longitude_1km']), np.ravel(data['Latitude_1km']), \
                                      bins=[bins['lon'], bins['lat']], weights=np.ravel(reflectance_ocean))[0]

    Lat_temp = np.histogram2d(data['Longitude_1km'][mask], data['Latitude_1km'][mask], \
                                   bins=[bins['lon'], bins['lat']], weights=data['Latitude_1km'][mask])[0]

    Lon_temp = np.histogram2d(data['Longitude_1km'][mask], data['Latitude_1km'][mask], \
                                   bins=[bins['lon'], bins['lat']], weights=data['Longitude_1km'][mask])[0]


    # Threshold for whole 100x100 km region - whether the whole region should or should not be included in L3
    # indexing data ranges to be used to calculate percentage of cloud properties in the region
    # (e.g., percentage of multi layer clouds, percentage of cloud top temperature etc.
    cld_Temperature_temp = np.zeros(np.shape(data['cloud_top_temperature_1km']))
    cld_Temperature_temp[np.isnan(data['cloud_top_temperature_1km'])] = 1
    cld_Temperature_temp[data['cloud_top_temperature_1km'] > 273] = 1
    cld_Temperature_temp_flag = np.histogram2d(np.ravel(data['Longitude_1km']), np.ravel(data['Latitude_1km']), \
                               bins=[bins['lon'], bins['lat']], weights=np.ravel(cld_Temperature_temp))[0]

    Multi_temp = np.zeros(np.shape(data['Cloud_Multi_Layer_Flag']))
    Multi_temp[np.isnan(data['Cloud_Multi_Layer_Flag'])] = 1 # NaN is no retrieval
    Multi_temp[data['Cloud_Multi_Layer_Flag'] == 1] = 1
    Multi_layer_flag = np.histogram2d(np.ravel(data['Longitude_1km']), np.ravel(data['Latitude_1km']), \
                                      bins=[bins['lon'], bins['lat']], weights=np.ravel(Multi_temp))[0]

    Cld_phase_temp = copy.copy(data['Cloud_Phase_Optical_Properties'])
    Cld_phase_temp[Cld_phase_temp <= 2] = 1
    Cld_phase_temp[Cld_phase_temp > 2] = 0
    Cld_phase_temp[np.isnan(Cld_phase_temp)] = 1
    Cld_phase_flag = np.histogram2d(np.ravel(data['Longitude_1km']), np.ravel(data['Latitude_1km']), \
                                    bins=[bins['lon'], bins['lat']], weights=np.ravel(Cld_phase_temp))[0]

    cld_mask_temp = np.ones(np.shape(data['Cloud_Mask_1km'][:, :, 0]))
    cld_mask_temp[data['Cloud_Mask_1km'][:,:,0]<=0] = 1
    cld_mask_flag = np.histogram2d(np.ravel(data['Longitude_1km']), np.ravel(data['Latitude_1km']), \
                   bins=[bins['lon'], bins['lat']], weights=np.ravel(cld_mask_temp))[0]

    # Masks whole 100x100 km region -  whether the whole region should or should not be included in L3
    #     # (e.g., percentage of multi layer clouds, percentage of cloud top temperature etc.
    Masks = [#(LTS < 18.55)*
             ((Multi_layer_flag / Num_temp_all) > 0.99)* # Take 1s
             ((Cld_phase_flag / Num_temp_all) > 0.99)* # Take 1s
             ((cld_Temperature_temp_flag / Num_temp_all)  > 0.99)*
             ((cld_mask_flag / Num_temp_all) > 0.99)][0]
    Masks = ~Masks

    CDNC_temp[Masks] = 0
    CDNC_temp[np.isnan(CDNC_temp)] = 0

    LWP_temp[Masks] = 0
    LWP_temp[np.isnan(LWP_temp)] = 0

    LWP_squared_temp[Masks] = 0
    LWP_squared_temp[np.isnan(LWP_squared_temp)] = 0

    reff_temp[Masks] = 0
    reff_temp[np.isnan(reff_temp)] = 0

    tau_temp[Masks] = 0
    tau_temp[np.isnan(tau_temp)] = 0

    CF_MODIS_temp[Masks] = 0
    CF_MODIS_temp[np.isnan(CF_MODIS_temp)] = 0

    delta_Dc_temp[Masks] = 0
    delta_Dc_temp[np.isnan(delta_Dc_temp)] = 0

    reflectance_temp[Masks] = 0
    reflectance_temp[np.isnan(reflectance_temp)] = 0

    Num_temp[Masks] = 0
    Num_temp[np.isnan(Num_temp)] = 0

    CF_tau_temp[Masks] = 0
    CF_tau_temp[np.isnan(Num_temp)] = 0

    Num_temp_all[Masks] = 0
    Num_temp_all[np.isnan(Num_temp_all)] = 0

    # python list -> np.concatenate (l_dictkeysappend in dlist,py) in the misc library l_toarray
    output_array['Num'] = np.append(output_array['Num'],Num_temp[Num_temp>0])
    output_array['CF_tau'] = np.append(output_array['CF_tau'],CF_tau_temp[Num_temp>0])
    output_array['Num_all'] = np.append(output_array['Num_all'], Num_temp_all[Num_temp>0])
    output_array['CDNC'] = np.append(output_array['CDNC'], CDNC_temp[Num_temp>0])
    output_array['LWP'] = np.append(output_array['LWP'], LWP_temp[Num_temp>0])
    output_array['LWP_squared'] = np.append(output_array['LWP_squared'], LWP_squared_temp[Num_temp>0])
    output_array['reff'] = np.append(output_array['reff'], reff_temp[Num_temp>0])
    output_array['tau'] = np.append(output_array['tau'], tau_temp[Num_temp>0])
    output_array['delta_Dc'] = np.append(output_array['delta_Dc'], delta_Dc_temp[Num_temp>0])
    output_array['CF_MODIS'] = np.append(output_array['CF_MODIS'], CF_MODIS_temp[Num_temp>0])
    output_array['reflectance'] = np.append(output_array['reflectance'], reflectance_temp[Num_temp>0])
    output_array['Lon'] = np.append(output_array['Lon'], Lon_temp[Num_temp > 0])
    output_array['Lat'] = np.append(output_array['Lat'], Lat_temp[Num_temp > 0])

    print('Done')

    return output_array

def creat_output_dict():
    lat = 360
    lon = 180
    output={}
    output['delta_Dc'] = np.zeros((lat, lon))
    output['count_num'] = np.zeros((lat, lon))
    output['Num'] = np.zeros((lat, lon))
    output['CF_tau'] = np.zeros((lat, lon))
    output['Num_all'] = np.zeros((lat, lon))
    output['LWP'] = np.zeros((lat, lon))
    output['LWP_nomask'] = np.zeros((lat, lon))
    output['LWP_squared'] = np.zeros((lat, lon))
    output['reff'] = np.zeros((lat, lon))
    output['tau'] = np.zeros((lat, lon))
    output['CF_MODIS'] = np.zeros((lat, lon))
    output['Multi_layer_flag'] = np.zeros((lat, lon))
    output['reflectance'] = np.zeros((lat, lon))
    output['CDNC'] = np.zeros((lat, lon))
    output['Num_all_pixels'] = np.zeros((lat, lon))
    output['Num_tau_pixels'] = np.zeros((lat, lon))
    output['Lon'] = np.zeros((lat, lon))
    output['Lat'] = np.zeros((lat, lon))
    return output

def calculate_Nd_adjust(re, tau, T=None, P=None):
    # calculating Cw based on T and P - more accurate than standard approaches
    # Input: effective radius [m] and tau [unitless]; Temperature and pressure are optional

    # Temperature and pressure if not given:
    T = T or 275 #[K]
    P = P or 95000 #[Pa]

    Qext = 2    # [unitless]
    ro_w = 997*10**3     #[gr*m^-3]

    Cp = 1004 # [J/kg K]
    ro_a = 1.2 # air density [kg/m3]
    Lv =  2.5 * 10**6 # latent heat of vaporization [J/kg]
    gamma_d = 9.81/Cp
    f_ad = 0.8

    Cw = f_ad * ((ro_a * Cp * (gamma_d - sat_lapse_rate(T, P)) / (Lv)) * 1000)  # eq 14 in Grosvenor 2018 [gr*m^-4]
    gamma = ((5**0.5)/(2*np.pi*0.8)) * (Cw/(Qext*ro_w))**0.5

    N = (gamma * tau ** 0.5 * (re * (1e-6)) ** (-5. / 2)) * 1e-6

    return(N)
