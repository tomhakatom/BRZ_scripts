
import numpy as np
import copy
import glob
import time
import create_L3
import read_data

########################################################################################################################
## Global analysis creating L3 data
########################################################################################################################
all_data = {}
file_list = np.sort(glob.glob('/media/tgoren/TOSHIBA EXT/NOAA/data/MODIS/MODIS_L2/SEP/*.*'))
modis_vars = ['Cloud_Effective_Radius', 'Cloud_Water_Path', 'Cloud_Optical_Thickness', \
              'cloud_top_temperature_1km', 'Cloud_Multi_Layer_Flag', \
              'Cloud_Water_Path_Uncertainty', 'Cloud_Effective_Radius_16', \
              'Cloud_Optical_Thickness_Uncertainty', 'Cloud_Effective_Radius_Uncertainty', \
              'Atm_Corr_Refl', 'Cloud_Fraction', 'Latitude', 'Longitude', \
              'Cloud_Mask_1km', 'Retrieval_Failure_Metric','Cloud_Top_Height',\
              'Solar_Zenith', 'Cloud_Phase_Optical_Properties']
ERA5_dir = '/home/tgoren/research/analysis/Global_CRE_Dc/data/ECMWF/'

lat =360
lon=180
factor = 1 #  fraction of observed Nd (droplet concentration); range: 0-1
CF_threshold = 0.1 # Exclude scenes with CF<CF_threshold

#output_all = create_L3.creat_output_dict()

output_array = {}
output_array['delta_Dc'] = np.array([])
output_array['count_num'] = np.array([])
output_array['Num'] = np.array([])
output_array['Num_land'] = np.array([])
output_array['Num_all'] = np.array([])
output_array['LWP'] = np.array([])
output_array['LWP_nomask'] = np.array([])
output_array['LWP_squared'] = np.array([])
output_array['reff'] = np.array([])
output_array['tau'] = np.array([])
output_array['CF_tau'] = np.array([])
output_array['CF_MODIS'] = np.array([])
output_array['Multi_layer_flag'] = np.array([])
output_array['reflectance'] = np.array([])
output_array['CDNC'] = np.array([])
output_array['Num_all_pixels'] = np.array([])
output_array['Num_tau_pixels'] = np.array([])
output_array['Lon'] = np.array([])
output_array['Lat'] = np.array([])

#run_indx = 1
for count, f_name_modis in enumerate(file_list):
    print(f_name_modis)
    modis_data_all = read_data.read_modis(f_name_modis, modis_vars) # read modis data
    modis_data = copy.copy(read_data.shorten_modis_swath_data(modis_data_all, 500, 500)) # Chossing data +- 500 km along ground track to avoid lower resolution and associated biases
    output_array = create_L3.grid_data(modis_data, f_name_modis, ERA5_dir, output_array, factor)
    print(count)
    #output = create_L3.creat_output_dict()
    #run_indx = run_indx+1

# Time counter - end
#elapsed = time.time() - t
#print(elapsed/60)