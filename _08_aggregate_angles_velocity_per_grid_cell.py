# !/usr/bin/env python.
# -*- coding: utf-8 -*-

"""
Read the hydraulic model grid file

Create a new grid in a way that nodes of first grid are center of second one
For every cell find all values that fall with in the cell,
Calculate the average fish velocity and fish swimming direction
per grid cell (sum values/ sum of points) 

Plot the results:
Every grid cell the average coordinates 
Average fish velocity and angle value
"""

__author__ = "Abbas El Hachem"
__copyright__ = 'Institut fuer Wasser- und Umweltsystemmodellierung - IWS'
__email__ = "abbas.el-hachem@iws.uni-stuttgart.de"

# =============================================================================

import timeit
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.ticker import FormatStrFormatter

from _00_define_main_directories import asci_grd_file_1m_
from _02_filter_fish_data_based_on_HPE_Vel_RMSE import (wgs82, utm32,
                                                        calculate_fish_velocity,
                                                        convert_coords_fr_wgs84_to_utm32_)

from _03_plot_margingals_histograms_velocity_hpe_rmse import fontsize, labelsize
from _03_plot_margingals_histograms_velocity_hpe_rmse import savefig
from _03_plot_margingals_histograms_velocity_hpe_rmse import out_save_dir

from _04_plot_heatmaps import rvb
from _07_calculate_angle_between_fish_positions_ import calculate_angle_between_two_positions

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

#==============================================================================
#
#==============================================================================


def plot_agg_grid_vls(grdx, grdy, var_vls, fish_nbr, var_name, out_plots_dir):
    '''
        a function to plot average velocity or angle per grid cell
    '''
    fig, ax0 = plt.subplots(1, 1, figsize=(20, 10), dpi=100)
    if var_name == 'Velocity':
        vmin, vmax, unit, extend = 0, 1.51, 'm/s', 'neither'
        ticks = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]

    if var_name == 'fish_angle':
        vmin, vmax, unit, extend = -180, 180, 'deg', 'neither'
        ticks = [-180, -135, -90, -45, 0, 45, 90, 135, 180]
    try:
        im = ax0.scatter(grdx, grdy, c=var_vls, s=.75, cmap=rvb, marker=',',
                         vmin=vmin, vmax=vmax, alpha=0.95)
    except ValueError:
        print('adjusting shapes')
        if grdx.shape[0] > var_vls.shape[0]:
            grdx = grdx[1:]
            grdy = grdy[1:]
        if var_vls.shape[0] > grdx.shape[0]:
            var_vls = var_vls[1:]
        im = ax0.scatter(grdx, grdy, c=var_vls, s=0.75, cmap=rvb, marker='s',
                         vmin=vmin, vmax=vmax, alpha=0.95)

    cbar = fig.colorbar(im, ax=ax0, extend=extend, fraction=0.024, pad=0.02,
                        ticks=ticks, boundaries=ticks, aspect=30)
    cbar.ax.set_ylabel('Mean %s %s' % (var_name, unit), fontsize=fontsize)
    cbar.ax.tick_params(labelsize=10)
    ax0.set_title('mean_%s_per_grid_cell_%s' % (var_name, fish_nbr),
                  fontsize=fontsize)
    ax0.set_xlim([10.222, 10.228]), ax0.set_ylim([47.8175, 47.8205])
    ax0.set_xticks([10.222, 10.223, 10.224, 10.225, 10.226, 10.227, 10.228])
    ax0.set_yticks([47.8175, 47.8180, 47.8185, 47.8190, 47.8195,
                    47.8200, 47.8205])
    ax0.set_xlabel('Longitude', fontsize=fontsize)
    ax0.set_ylabel('Latitude', fontsize=fontsize)
    ax0.tick_params(axis='x', labelsize=labelsize)
    ax0.tick_params(axis='y', labelsize=labelsize)

    ax0.xaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    ax0.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    savefig('mean_%s_per_grid_cell_%s' % (var_name, fish_nbr),
            out_plots_dir)
    plt.close(fig)
    pass


# =============================================================================
#
# =============================================================================
if __name__ == '__main__':

    print('**** Started on %s ****\n' % time.asctime())
    START = timeit.default_timer()  # to get the runtime of the program
    # call function here
    STOP = timeit.default_timer()  # Ending time
    print(('\n****Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ***' % (time.asctime(), STOP - START)))
