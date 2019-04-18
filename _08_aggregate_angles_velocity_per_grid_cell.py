# !/usr/bin/env python.
# -*- coding: utf-8 -*-

"""Gets and prints the spreadsheet's header columns

Parameters
----------
file_loc : str
    The file location of the spreadsheet
print_cols : bool, optional
    A flag used to print the columns to the console (default is False)

Returns
-------
list
    a list of strings representing the header columns
"""

__author__ = "Abbas El Hachem"
__copyright__ = 'Institut fuer Wasser- und Umweltsystemmodellierung - IWS'
__email__ = "abbas.el-hachem@iws.uni-stuttgart.de"

# ===================================================

from _00_define_main_directories import asci_grd_file_1m_
from _02_filter_fish_data_based_on_HPE_Vel_RMSE import (wgs82, utm32,
                                                        calculate_fish_velocity,
                                                        convert_coords_fr_wgs84_to_utm32_)
from _07_calculate_angle_between_fish_positions_ import calculate_angle_between_two_positions
import os
import timeit
import time
import numpy as np
import pandas as pd

#==============================================================================
#
#==============================================================================


def aggregate_values_per_grid_cell(df_fish, asci_grd_file,
                                   vel_col_name='Fish_swim_velocity_in_m_per_s',
                                   angle_col_name='fish_swim_direction_compared_to_x_axis'):
    '''
        a function to aggregate values per grid cell
        create a grid sikilar to the flow grid
        but in a way that the center of this grid are
        the nods of the flow grid
        calculate the number of fish positions
        in a cell, find average velocity and angle
    '''
    if 'Velocity' not in df_fish.columns:
        df_fish = calculate_fish_velocity(df_fish)
    if 'fish_swim_direction_compared_to_x_axis' not in df_fish.columns:
        df_fish = calculate_angle_between_two_positions(df_fish)
    grid_spacing = 1
    asci_grid = pd.read_csv(asci_grd_file, sep=',')

    xmin, xmax = asci_grid.x.min() - 0.5, asci_grid.x.max() + 0.5
    ymin, ymax = asci_grid.y.min() - 0.5, asci_grid.y.max() + 0.5
    x_vals = np.arange(xmin, xmax + 0.1, grid_spacing)
    y_vals = np.arange(ymin, ymax + 0.1, grid_spacing)
    x_fish, y_fish = convert_coords_fr_wgs84_to_utm32_(
        wgs82, utm32, df_fish.Longitude.values, df_fish.Latitude.values)

    fish_cols = [int((x - xmin) / grid_spacing) for x in x_fish]
    fish_rows = [int((y - ymin) / grid_spacing) for y in y_fish]

    fish_coords_x = np.zeros((y_vals.shape[0], x_vals.shape[0]))
    fish_coords_y = np.zeros((y_vals.shape[0], x_vals.shape[0]))
    mean_angle_grid = np.zeros((y_vals.shape[0], x_vals.shape[0]))
    sum_velo_grid = np.zeros((y_vals.shape[0], x_vals.shape[0]))
    freq_grid = np.zeros((y_vals.shape[0], x_vals.shape[0]), dtype=int)

    for i, (fish_row, fish_col) in enumerate(zip(fish_rows, fish_cols)):
        fish_coords_x[fish_row, fish_col] += df_fish.iloc[i].loc['Longitude']
        fish_coords_y[fish_row, fish_col] += df_fish.iloc[i].loc['Latitude']

        sum_velo_grid[fish_row, fish_col] += df_fish.iloc[i].loc[vel_col_name]
        mean_angle_grid[fish_row, fish_col] += df_fish.iloc[i].loc[
            angle_col_name]
        freq_grid[fish_row, fish_col] += 1

    mean_x_grid = fish_coords_x / freq_grid
    mean_y_grid = fish_coords_y / freq_grid
    mean_vel_grid = sum_velo_grid / freq_grid
    mean_dir_grid = mean_angle_grid / freq_grid

    grdx = mean_x_grid[~np.isnan(mean_x_grid)]
    grdy = mean_y_grid[~np.isnan(mean_y_grid)]
    vel_vls = mean_vel_grid[~np.isnan(mean_vel_grid)]
    angle_vls = mean_dir_grid[~np.isnan(mean_dir_grid)]
    return vel_vls, angle_vls, grdx, grdy


if __name__ == '__main__':

    print('**** Started on %s ****\n' % time.asctime())
    START = timeit.default_timer()  # to get the runtime of the program

    STOP = timeit.default_timer()  # Ending time
    print(('\n****Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ***' % (time.asctime(), STOP - START)))
