# !/usr/bin/env python.
# -*- coding: utf-8 -*-

"""
Calculate the swim direction between each two successive fish locations
Add the result to the dataframe, in a new column

fish_file[-10:-5] refers to the fish number (id) maybe need to change it
depending on the file name
"""

__author__ = "Abbas El Hachem"
__copyright__ = 'Institut fuer Wasser- und Umweltsystemmodellierung - IWS'
__email__ = "abbas.el-hachem@iws.uni-stuttgart.de"

# ===================================================

import os
import timeit
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from _00_define_main_directories import (dir_kmz_for_fish_names,
                                         out_data_dir, img_loc)

from _01_filter_fish_points_keep_only_in_river import getFiles
from _02_filter_fish_data_based_on_HPE_Vel_RMSE import filtered_out_data
from _03_plot_margingals_histograms_velocity_hpe_rmse import (plot_img,
                                                              savefig,
                                                              fontsize,
                                                              labelsize)
from _04_plot_heatmaps import rvb

#==============================================================================
# # def all directories and all required parameters
#==============================================================================

out_save_dir = os.path.join(out_data_dir, r'Df_filtered_with_angles')
if not os.path.exists(out_save_dir):
    os.mkdir(out_save_dir)

#==============================================================================
#
#==============================================================================


def calculate_angle_between_two_positions(df_fish, xname='Longitude',
                                          yname='Latitude'):
    ''' calculate angle between two successive positions '''

    x_vals, y_vals = df_fish[xname].values, df_fish[yname].values

    angles_degs = [np.math.degrees(np.math.atan2(y_vals[i] - y_vals[i - 1],
                                                 x_vals[i] - x_vals[i - 1]))
                   for i in range(1, df_fish.values.shape[0])]
    angles_degs.insert(0, np.nan)
    df_fish['fish_swim_direction_compared_to_x_axis'] = angles_degs
    return df_fish


#==============================================================================
#
#==============================================================================

def plot_angles_two_positions(lons, lats, angles, fish_type_nbr, out_plots_dir):
    ''' fct to plot the angle between two consecutive locations'''
    _, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=800)
    ax.quiver(lons, lats, np.cos(angles), np.sin(angles), cmap=rvb,
              pivot='mid', width=0.00112, alpha=0.05,
              scale=1. / .010505)

    ax.scatter(lons, lats, color='b', s=0.5, alpha=0.05, marker=',')

    plot_img(img_loc, ax)
    ax.set_xlabel('Longitude', fontsize=fontsize)
    ax.set_ylabel('Latitude', fontsize=fontsize)
    ax.tick_params(axis='x', labelsize=labelsize)
    ax.tick_params(axis='y', labelsize=labelsize)
    ax.set_title('angles_vectors_fish: %s' % (fish_type_nbr),
                 fontsize=fontsize, y=0.99)
    savefig('%s_fish_%s' % ('angles_vectors_fish_', fish_type_nbr),
            out_plots_dir)
    plt.clf()
    plt.close()
    return


#==============================================================================
#
#==============================================================================
if __name__ == '__main__':

    print('**** Started on %s ****\n' % time.asctime())
    START = timeit.default_timer()  # to get the runtime of the program

    in_filtered_fish_files_dict = getFiles(filtered_out_data, '.csv',
                                           dir_kmz_for_fish_names)

    for fish_type in in_filtered_fish_files_dict.keys():
        for fish_file in in_filtered_fish_files_dict[fish_type]:
            print(fish_file)

            fish_type_nbr = fish_type + '_' + fish_file[-10:-5]
            try:
                in_df_fish = pd.read_csv(fish_file, index_col=0, sep=',')
                df_fish_agl = calculate_angle_between_two_positions(in_df_fish)
            except Exception:
                raise Exception
            break
        break

    STOP = timeit.default_timer()  # Ending time
    print(('\n****Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ***' % (time.asctime(), STOP - START)))
