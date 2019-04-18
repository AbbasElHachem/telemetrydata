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
__copyright__ = 'Institut f�r Wasser- und Umweltsystemmodellierung - IWS'
__email__ = "abbas.el-hachem@iws.uni-stuttgart.de"

# ===================================================

from _00_define_main_directories import img_loc

from _03_plot_margingals_histograms_velocity_hpe_rmse import (savefig,
                                                              plot_img,
                                                              extent,
                                                              fontsize,
                                                              labelsize)

from _04_plot_heatmaps import rvb

import os
import timeit
import time

import matplotlib.pyplot as plt
import pandas as pd

import matplotlib.colors as mcolors
import numpy as np


# =============================================================================
#
# =============================================================================


def plot_loc_time_vls(df_fish, fish_type_nbr, out_plots_dir):
    ''' fct to plot Time difference between each 2 locations '''
    int_time = [(df_fish.index[i] - df_fish.index[i - 1]
                 ) // pd.Timedelta('1s') for i, _ in enumerate(df_fish.index)]
    df_fish['Time Epoch'] = int_time

    df_fish_below_5s = df_fish[(df_fish['Time Epoch'].values <= 5)
                               & (df_fish['Time Epoch'].values > 0)]
    df_fish_abv_5s = df_fish[df_fish['Time Epoch'].values > 5]
    var_bounds = np.linspace(0, 5, 5, endpoint=True)
    norm = mcolors.BoundaryNorm(boundaries=var_bounds, ncolors=256)
    fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=800)

    ax.scatter(df_fish_abv_5s['Longitude'].values,
               df_fish_abv_5s['Latitude'].values,
               c='k', s=0.05, alpha=0.25, marker='+',
               label='Time Difference > 5s')

    plot_img(img_loc, ax)
    pcm = ax.scatter(df_fish_below_5s['Longitude'].values,
                     df_fish_below_5s['Latitude'].values,
                     c=df_fish_below_5s['Time Epoch'].values,
                     s=0.05, alpha=0.05, marker=',',
                     cmap=rvb, norm=norm,
                     vmin=df_fish['Time Epoch'].values.min(),
                     vmax=df_fish['Time Epoch'].values.max(),
                     label='Time Difference <= 5s')

    ax.set_xlim(extent[0], extent[1]), ax.set_ylim(extent[2], extent[3])
    cb = fig.colorbar(pcm, ax=ax, extend='max', orientation='vertical')
    cb.ax.tick_params(labelsize=labelsize)
    cb.ax.set_ylabel('Time Difference values (s)', fontsize=fontsize)
    cb.set_alpha(1), cb.draw_all()
    ax.set_xlabel('Longitude', fontsize=fontsize)
    ax.set_ylabel('Latitude', fontsize=fontsize)
    ax.tick_params(axis='x', labelsize=labelsize)
    ax.tick_params(axis='y', labelsize=labelsize)
    ax.set_title('Time Difference values above 5 (s) Values for Fish: %s'
                 % (fish_type_nbr), fontsize=fontsize, y=0.99)
    plt.legend(loc=0, fontsize=fontsize)
    plt.grid(alpha=0.5)
    savefig('%s_fish_%s'
            % ('Time Difference values above (5s)', fish_type_nbr),
            out_plots_dir)

    plt.close()
    df_fish.to_csv(os.path.join(out_plots_dir, 'fish_nbr_%s_with_time_diff.csv'
                                % fish_type_nbr))
    return
# =============================================================================
#
# =============================================================================


if __name__ == '__main__':

    print('**** Started on %s ****\n' % time.asctime())
    START = timeit.default_timer()  # to get the runtime of the program
    # plot_loc_time_vls()
    STOP = timeit.default_timer()  # Ending time
    print(('\n****Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ***' % (time.asctime(), STOP - START)))
