# !/usr/bin/env python.
# -*- coding: utf-8 -*-

"""
Read the intersected fish-river data

Filter the data based on the HPE-RMSE simultaneous
thresholds derived from the statistics done to keep
90% of the data with an approximate error of 1m

Calculate Velocities between consecutive Fish positions
and remove all first positions of velocities above 1.5 m/s

"""

__author__ = "Abbas El Hachem"
__copyright__ = 'Institut fuer Wasser- und Umweltsystemmodellierung - IWS'
__email__ = "abbas.el-hachem@iws.uni-stuttgart.de"

# =============================================================================
from __future__ import division

from _00_define_main_directories import (dir_kmz_for_fish_names,
                                         main_data_dir,
                                         out_data_dir)

from _01_filter_fish_points_keep_only_in_river import getFiles
import os

import pyproj
import numpy as np
import pandas as pd

# def epsg wgs84 and utm32
wgs82 = "+init=EPSG:4326"
utm32 = "+init=EPSG:32632"

# def HPE, RMSE, Velocity thresholds for filtering
hpe_thr = 1.35
rmse_thr = 0.35
vel_thr = 1.5

#==============================================================================
#
#==============================================================================

filtered_out_data = os.path.join(out_data_dir, r'Filtered_df_HPE_RMSE_VEL')

# if not os.path.exists(filtered_out_data):
#    os.mkdir(filtered_out_data)


#==============================================================================
#
#==============================================================================
def readDf(df_file):
    ''' read on df and adjust index and selct columns'''
    df = pd.read_csv(df_file, sep=',', index_col=0, infer_datetime_format=True,
                     usecols=['Time', 'Longitude', 'Latitude', 'HPE', 'RMSE'])
    time_fmt = '%Y-%m-%d %H:%M:%S.%f'
    try:
        df.index = pd.to_datetime(df.index, format=time_fmt)
    except ValueError:
        df.index = [ix.replace('.:', '.') for ix in df.index]
        df.index = pd.to_datetime(df.index, format=time_fmt)
    return df

# In[67]:


def convert_coords_fr_wgs84_to_utm32_(epgs_initial_str, epsg_final_str,
                                      first_coord, second_coord):
    '''fct to convert points from wgs 84 to utm32'''
    initial_epsg = pyproj.Proj(epgs_initial_str)
    final_epsg = pyproj.Proj(epsg_final_str)
    x, y = pyproj.transform(initial_epsg, final_epsg,
                            first_coord, second_coord)
    return x, y

# In[68]:


def calculate_distance_2_points(deltax, deltay):
    ''' fct to calculate distance between two arrays of coordinates'''
    return np.sqrt((deltax ** 2) + (deltay ** 2))

# In[104]:


def calculate_fish_velocity(df_fish, wgs82, utm32):
    ''' function to calculate travel velocity between subsequent locations'''

    x, y = convert_coords_fr_wgs84_to_utm32_(wgs82, utm32,
                                             df_fish['Longitude'].values,
                                             df_fish['Latitude'].values)

    df_utm32 = pd.DataFrame(index=df_fish.index,
                            data={'delta_x': x, 'delta_y': y})

    delta_x_y = df_utm32.diff()
    delta_x_y.dropna(inplace=True)
    delta_x_y['Time'] = delta_x_y.index

    (delta_x_y['Longitude'], delta_x_y['Latitude']) = (
        convert_coords_fr_wgs84_to_utm32_(utm32, wgs82, x[1:], y[1:]))

    delta_x_y['Fish_x_coord'], delta_x_y['Fish_y_coord'] = x[1:], y[1:]

    delta_x_y['Time_Difference_in_s'] = np.round(
        delta_x_y.Time.diff() / pd.Timedelta('1s'), 2)

    delta_x_y['Traveled_Distance_in_m'] = np.round(
        calculate_distance_2_points(delta_x_y['delta_x'],
                                    delta_x_y['delta_y']), 2)

    delta_x_y['Fish_swim_velocity_in_m_per_s'] = np.round(
        (delta_x_y['Traveled_Distance_in_m'].values /
         delta_x_y['Time_Difference_in_s'].values), 2)

    delta_x_y.drop(columns=['Time', 'delta_x', 'delta_y'],
                   axis=1, inplace=True)

    delta_x_y['HPE'], delta_x_y['RMSE'] = np.round(
        (df_fish['HPE'].values[1:], df_fish['RMSE'].values[1:]), 2)
    return delta_x_y

# In[105]:


def use_Variable_below_thr_keep_first_point(df_fish, var_name, var_thr):
    ''' use a filter based on Variable threshold keep first point'''
    df_fish = df_fish.copy()
    df_fish = df_fish[df_fish[var_name] <= var_thr]
    df_fish = df_fish[0 < df_fish[var_name]]
    return df_fish

# In[106]:


def use_Variable_below_thr_two_var(df_fish, var1_name, var1_thr,
                                   var2_name, var2_thr):
    ''' use a filter based on two different Variables thresholds'''
    df_fish = df_fish.copy()
    df_fish = df_fish[(df_fish[var1_name] <= var1_thr) &
                      (df_fish[var2_name] <= var2_thr)]
    return df_fish

# In[107]:


def use_var_blw_thr_keep_second_point(df_fish, var_name, var_thr):
    ''' use a filter based on Variable threshold keep second point'''
    df_fish_red = df_fish.copy()
    # tag rows based on the threshold
    df_fish_red['avg_thr'] = df_fish_red[var_name] > var_thr
    # this is previous idx of where values are abv threshold
    ix = df_fish_red.index[df_fish_red['avg_thr'].shift(-1).fillna(False)]

    # drop this index of dataframe
    # in this way we keep the second point and drop first point
    df_fish_red.drop(index=ix, inplace=True)
    return df_fish_red

# In[108]:


def drop_vals_blw_thr_keep_pts_based_on_other_var(df_fish,
                                                  var_name='Velocity',
                                                  var_thr=1.5,
                                                  othr_var1='HPE',
                                                  othr_var2='RMSE'):
    '''
        a fct used to remove values abv a threshold based on
        values of other variables. In this case, this is used
        to remove the velocity values that are above the threshold
        based on the values of HPE and RMSE
    '''
    df_fish = df_fish.copy()
    df_fish['avg_thr'] = df_fish[var_name] > var_thr

    # this is previous idx of where values are abv threshold
    ixb4 = df_fish.index[df_fish['avg_thr'].shift(-1).fillna(False)]
    # idx of where vls are abv thr
    ixafter = df_fish.index[df_fish['avg_thr'].fillna(False)]
    result = [None] * (len(ixb4) + len(ixafter))

    result[::2] = ixb4
    result[1::2] = ixafter

    ix_keep = []
    for ix0, ix1 in zip(ixb4, ixafter):
        hpe0, rmse0 = df_fish.loc[ix0, othr_var1], df_fish.loc[ix0, othr_var2]
        hpe1, rmse1 = df_fish.loc[ix1, othr_var1], df_fish.loc[ix1, othr_var2]
        if (hpe0 < hpe1) and (rmse0 < rmse1):
            ix_keep.append(ix1)
        if (hpe0 > hpe1) and(rmse0 > rmse1):
            ix_keep.append(ix0)
        else:
            ix_keep.append(-99)
    return result, ixb4, ixafter

# In[109]:


def use_Variable_abv_thr(df_fish, var_name, var_thr):
    ''' use a filter based on Variable threshold'''
    df_fish = df_fish.copy()
    df_fish = df_fish[df_fish[var_name] >= var_thr]
    return df_fish

# In[ ]:


if __name__ == '__main__':

    in_fish_files_dict = getFiles(
        main_data_dir, '.csv', dir_kmz_for_fish_names)

    for fish_type in in_fish_files_dict.keys():
        for fish_file in in_fish_files_dict[fish_type]:
            print(fish_file)

            fish_nbr = fish_type + '_' + fish_file[-9:-4]
            try:
                df_orig = readDf(fish_file)
                delta_x_y_red = use_Variable_below_thr_two_var(df_orig,
                                                               'HPE',
                                                               hpe_thr,
                                                               'RMSE',
                                                               rmse_thr)
                delta_x_y = calculate_fish_velocity(
                    delta_x_y_red, wgs82, utm32)
                delta_x_y = use_Variable_below_thr_keep_first_point(
                    delta_x_y, 'Fish_Swimming_Velocity_in_m_per_s', 1.5)
                delta_x_y.to_csv(
                    os.path.join(filtered_out_data,
                                 'filtered_data_' + fish_nbr + '_.csv'))
            except Exception as msg:
                print(msg)
                continue
            # break
        # break

# In[ ]:
