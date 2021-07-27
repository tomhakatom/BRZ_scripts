import numpy as np
import scipy
import netCDF4 as nc4
import datetime
from math import radians, cos, sin, asin, sqrt
import matplotlib.pyplot as plt
import time

def lon_180_to_360(lon):
    # Change -180 to +180 to 0 to 360
    if lon.size > 1:
        lon[np.where(lon<=0)] = lon[np.where(lon<=0)] + 360
    else:
        if lon < 0:
            lon = lon +360
    return lon
    
def get_cf_tau(tau_box):
    # Calculate the cloud fraction based on retrieved tau pixels
    # Input: A matrix of tau
    # Output: CF
    mat_size = np.double(tau_box.size)
    tau_retrieved = np.count_nonzero(~np.isnan(tau_box))
    cf = (tau_retrieved/mat_size) * 100.0
    return cf

def mean_confidence_interval(data, confidence=0.95):
    # Compute confidence interval
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t._ppf((1+confidence)/2., n-1)
    return h

def regrid(lat, lon, data, lat_interval):
    # Re-grid data into lar/lon spaced by lat_interval
    # Input: lat, lon (1D); data (2D); lat_interval (int)
    # Output: the new gridded field
    new_grid_temp = {}
    data_arry = np.ravel(np.rot90(data))
    x, y = np.meshgrid(lat, lon)
    lat_original = np.ravel(x)
    lon_original = np.ravel(y)
    lat_new_grid = np.arange(-90,91,lat_interval)
    lon_new_grid = np.arange(-180, 181,lat_interval)
    new_grid_temp['Num'] = np.histogram2d(\
        lat_original, lon_original, bins=[lat_new_grid, lon_new_grid])[0].astype('int')
    new_grid_temp['data'] = np.histogram2d(\
        lat_original, lon_original, bins=[lat_new_grid, lon_new_grid], weights=data_arry)
    new_grid = new_grid_temp['data'][0]/ new_grid_temp['Num']
    return new_grid, lat_new_grid, lat_new_grid

def regrid_general(data, interval):
    # Re-grid data
    # Input: data (2D); interval (int)
    # Output: the new gridded field
    new_grid_temp = {}
    dims = np.shape(data)
    x, y = np.meshgrid(np.arange(dims[0]),np.arange(dims[1]))
    lat_original = np.ravel(x)
    lon_original = np.ravel(y)
    new_grid = np.arange(0, lat_original[-1], interval)
    new_grid = np.arange(0, lon_original[-1], interval)
    new_grid_temp['Num'] = np.histogram2d(\
        lat_original, lon_original, bins=[new_grid, new_grid])[0].astype('int')
    new_grid_temp['data'] = np.histogram2d(\
        lat_original, lon_original, bins=[new_grid, new_grid], weights=data_arry)
    new_grid = new_grid_temp['data'][0]/ new_grid_temp['Num']
    return new_grid
  
def get_time_since(time, st_year, st_month, st_day):
    # Input: start (days since) year, month, day; time is in [days]
    start_date = datetime.datetime(st_year, st_month, st_day,0,0,0)
    time_step = [start_date + datetime.timedelta(time[0])][0]
    return time_step

def get_date_time_since(year, days_since):
    # Return the date, given hours since
    # Input: year, days_since day 1 of the year (input 1)
    date = [datetime.timedelta(days_since) + datetime.datetime(year, 1, 1, 0, 0, 0)][0]
    return date

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def field_interpolate(data, factor):
    # Reducing resolution
    # Input: data field and factor (e.g., smaller by 2x2=2, 100x100 =100)

    resized_data = np.zeros([np.arange(0, data.shape[0], factor).shape[0],\
                      np.arange(0, data.shape[1], factor).shape[0]]) * np.nan

    for ii, i in enumerate(np.arange(0,data.shape[0], factor)):
        for jj, j in enumerate(np.arange(0, data.shape[1], factor)):
            resized_data[ii,jj] = np.nanmean(data[i:i+factor, j:j+factor])
    return resized_data

def field_interpolate_cores(tau, lwp, factor):
    # Reducing resolution
    # Input: data(tau, LWP) and factor (e.g., smaller by 2x2=2, 100x100 =100)

    resized_data = np.zeros([np.arange(0, lwp.shape[0], factor).shape[0],\
                      np.arange(0, lwp.shape[1], factor).shape[0]]) * np.nan

    for ii, i in enumerate(np.arange(0,lwp.shape[0], factor)):
        for jj, j in enumerate(np.arange(0, lwp.shape[1], factor)):
            tau_temp = tau[i:i+factor, j:j+factor]
            if np.size(tau_temp[~np.isnan(tau_temp)]) == 0:
                continue
            tau_precentile = np.percentile(tau_temp[~np.isnan(tau_temp)],85)
            if ~np.isnan(tau_precentile):
                resized_data[ii,jj] = np.nanmean(lwp[i:i+factor, j:j+factor][tau[i:i+factor, j:j+factor]>tau_precentile])
    return resized_data


def doy_to_date(year,doy):
    '''doy_to_date(year,doy)
    Converts a date from DOY representation to day.month.year
    returns tuple(year,month,day)
    Raises ValueError if the doy is not valid for that year'''
    dat = datetime.date(year,1,1)
    dat += datetime.timedelta(int(doy)-1)
    if dat.year != year:
        raise ValueError('Day not within year')
    return (dat.year,dat.month,dat.day)

def jdtodatestd (jdate):
    # Convert Julian date to regular date
    fmt = '%Y%j'
    datestd = datetime.datetime.strptime(jdate, fmt).date()
    return(datestd)


def date_to_doy(year,month,day):
    '''Converts date format from y,m,d to a tuple (year,doy)'''
    return (year,datetime.date(year,month,day).timetuple().tm_yday)

def get_LTS(temperature1000, temperature700):
    # calculate LTS
    # Input: Temperature at 1000 hpa, Temperature at 700 hpa (C)
    # Output: LTS
    R_cp = 0.286
    theta_700 = temperature700 * (1000/700.)**R_cp
    theta_1000 = temperature1000 * (1000/1000.)**R_cp
    LTS = theta_700 - theta_1000
    return LTS
