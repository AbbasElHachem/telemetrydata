# !/usr/bin/env python.
# -*- coding: utf-8 -*-

"""
Script used to plot the heatmaps of the filtered fish locations
"""

__author__ = "Abbas El Hachem"
__copyright__ = 'Institut fuer Wasser- und Umweltsystemmodellierung - IWS'
__email__ = "abbas.el-hachem@iws.uni-stuttgart.de"

# ===================================================

from matplotlib.ticker import FormatStrFormatter

from _00_define_main_directories import (dir_kmz_for_fish_names,
                                         out_data_dir,
                                         img_loc)
from _01_filter_fish_points_keep_only_in_river import getFiles
from _02_filter_fish_data_based_on_HPE_Vel_RMSE import filtered_out_data
from _03_plot_margingals_histograms_velocity_hpe_rmse import savefig, plot_img


import matplotlib as mpl
import os
import timeit
import time

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#==============================================================================
# # def all directories and all required parameters
#==============================================================================

out_save_dir = os.path.join(out_data_dir, r'Plots_Heatmaps')
if not os.path.exists(out_save_dir):
    os.mkdir(out_save_dir)

# def some parameters (no need to change)
# def extent of the river image for plotting
extent = [10.2210163765499988, 10.2303021853499985,
          47.8146222938500003, 47.8224152275500032]

# def font- and labelsize for plots
fontsize, labelsize = 10, 8

#==============================================================================
#
#==============================================================================


def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])

    return mcolors.LinearSegmentedColormap('CustomMap', cdict)

# =============================================================================
#
# =============================================================================


def calculate_weights_for_heatmaps(df_fish):
    '''
        function to calculate weights (time diff between 2 observations)
        to be used for heatmap
    '''
    int_time = [(df_fish.index[i] - df_fish.index[i - 1]
                 ) // pd.Timedelta('1s') for i, _ in enumerate(df_fish.index)]
    df_fish['Time Epoch'] = int_time
    df_fish.iloc[0, 11] = 0  # replace first entry by 0
    df_fish['Weights'] = df_fish.apply(lambda row: row['Time Epoch']
                                       if (row['distance'] <= 0.5)
                                       else 1, axis=1)
    return df_fish
# =============================================================================
#
# =============================================================================


def do_hist2d_for_heatmap(x, y, bins=1000, weights=None):
    ''' function to calculate the density for heatmaps '''
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins,
                                             normed=False, weights=weights)
    # heatmap = gaussian_filter(heatmap, sigma=600)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap.T, extent

# =============================================================================
#
# =============================================================================


def plot_heatmapt_fish_loc(df_fish, fish_type_nbr,
                           cmap_to_use,
                           out_plots_dir,
                           plt_img=False, weights=None):
    ''' fct to plot heatmap of a fish using time spent as weights if needed'''

    if weights is not None:
        df_fish = calculate_weights_for_heatmaps(df_fish)
        weights = df_fish['Weights'].values

    plt.rcParams['agg.path.chunksize'] = 10000

    x = df_fish['Longitude'].values
    y = df_fish['Latitude'].values
    fig, (ax1, ax2) = plt.subplots(
        1, 2, sharey=True, sharex=True,
        figsize=(20, 12), dpi=400)

    if plt_img:
        plot_img(img_loc, ax1), plot_img(img_loc, ax2)

    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax2.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    plt.subplots_adjust(wspace=0.05, hspace=0.15)

    ax1.tick_params(axis='x', labelsize=7)
    ax1.tick_params(axis='y', labelsize=7)
    ax1.grid(alpha=0.1)

    ax2.tick_params(axis='x', labelsize=7)
    ax2.tick_params(axis='y', labelsize=7)

    ax2.grid(alpha=0.1)
    ax1.scatter(x, y, c='darkblue', s=0.15,
                marker=',', alpha=0.15)
    ax1.set_title('Positions for fish %s '
                  % (fish_type_nbr), fontsize=fontsize)
    ax1.set_xlim([10.222, 10.228])
    ax1.set_ylim([47.8175, 47.8205])
    ax1.set_xticks([10.222, 10.223, 10.224, 10.225, 10.226, 10.227, 10.228])
    ax1.set_yticks([47.8175, 47.8180, 47.8185, 47.8190,
                    47.8195, 47.8200, 47.8205])

    ax2.set_xlim([10.222, 10.228]), ax2.set_ylim([47.8175, 47.8205])
    ax2.set_xticks([10.222, 10.223, 10.224, 10.225, 10.226, 10.227, 10.228])
    ax2.set_yticks([47.8175, 47.8180, 47.8185, 47.8190,
                    47.8195, 47.8200, 47.8205])

    img, extent_ht = do_hist2d_for_heatmap(x, y, weights=weights)
    max_tick = np.round(np.max(np.log(img)), 0)
    img[img == 0] = np.nan

    ax2.imshow(np.log(img), extent=extent_ht, origin='lower',
               cmap=cmap_to_use)
    ax2.set_title("Bi-dimensional Log Histogram Lon-Latt fish %s"
                  % (fish_type_nbr),
                  fontsize=fontsize)
    ticks = np.linspace(0, max_tick + 0.2, 7, endpoint=True)

    norm = mcolors.BoundaryNorm(ticks, cmap_to_use.N)
    ax_legend = fig.add_axes([0.4025, 0.20525, 0.5, 0.0125], zorder=3)
    cb = mpl.colorbar.ColorbarBase(ax_legend, ticks=ticks,  # extend='none',
                                   boundaries=ticks, norm=norm,
                                   cmap=cmap_to_use,
                                   orientation='horizontal')
    cb.set_label('Log Histogram Lon-Latt', fontsize=labelsize)
    cb.draw_all()
    cb.set_alpha(1)

    savefig('heatmap_fish_%s_' % (fish_type_nbr),
            out_plots_dir)
    plt.close()

    return
# =============================================================================
#
# =============================================================================


# make customized colormap
c = mcolors.ColorConverter().to_rgb
rvb = make_colormap([c('blue'), c('lightblue'), c('c'), 0.33,
                     c('green'), c('yellow'), c('gold'), 0.66,
                     c('orange'), c('red')])

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
                df_fish = pd.read_csv(fish_file, index_col=0, sep=',')
                plot_heatmapt_fish_loc(df_fish, fish_type_nbr, rvb,
                                       out_save_dir, plt_img=True,
                                       weights=None)
            except Exception:
                raise Exception
            break
        break
    STOP = timeit.default_timer()  # Ending time
    print(('\n****Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ***' % (time.asctime(), STOP - START)))
