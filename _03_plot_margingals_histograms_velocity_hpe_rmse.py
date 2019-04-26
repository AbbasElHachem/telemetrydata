#!/usr/bin/env python
# coding: utf-8

"""
Created on 15-03-2019

@author: EL Hachem Abbas,
Institut fuer Wasser- und Umweltsystemmodellierung - IWS
"""
from __future__ import division
from _00_define_main_directories import (dir_kmz_for_fish_names,
                                         orig_station_file,
                                         orig_data_dir,
                                         out_data_dir,
                                         img_loc)
from _01_filter_fish_points_keep_only_in_river import (getFiles, readDf)

from _02_filter_fish_data_based_on_HPE_Vel_RMSE import (calculate_fish_velocity,
                                                        filtered_out_data)

from PIL import Image
from matplotlib import style

from scipy import stats

import os

import matplotlib.colors as mcolors
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

Image.MAX_IMAGE_PIXELS = 1000000000
style.use('fivethirtyeight')

#==============================================================================
# # def what to plot
#==============================================================================
plot_vel_hpe_rmse_original_data = True
plot_vel_hpe_rmse_filtered_data = False

plot_original_histograms_and_marginals_vel_hpe_rmse_ = False
plot_filtered_histograms_and_marginals_vel_hpe_rmse_ = False

#==============================================================================
# # def all directories and all required parameters
#==============================================================================

out_save_dir = os.path.join(out_data_dir, r'Plots_HPE_RMSE_VEL')
if not os.path.exists(out_save_dir):
    os.mkdir(out_save_dir)

# def some parameters (no need to change)
# def extent of the river image for plotting
extent = [10.2210163765499988, 10.2303021853499985,
          47.8146222938500003, 47.8224152275500032]
# def epsg wgs84 and utm32
wgs82 = "+init=EPSG:4326"
utm32 = "+init=EPSG:32632"

# def font- and labelsize for plots
fontsize, labelsize = 10, 8

#==============================================================================
# START WRITTING FUNCTIONS HERE
#==============================================================================


def read_OrigStn_DF(df_stn_orig_file):
    ''' function to read fixed station data'''
    df_orig_stn = pd.read_csv(df_stn_orig_file, sep=',',
                              usecols=['StationName', 'Longitude', 'Latitude'])
    return df_orig_stn
#==============================================================================
#
#==============================================================================


def savefig(fig_name, out_dir):
    ''' fct to save fig based on name and outdir'''
    return plt.savefig(os.path.join(out_dir, '_%s.png' % fig_name),
                       frameon=True, papertype='a4',
                       bbox_inches='tight', pad_inches=.2)

#==============================================================================
#
#==============================================================================


def plot_img(img_loc, ax, img_transparancy=0.5):
    ''' fct to plot orginal river image
    img_loc :path to the Ortho image
    ax: matplotib suplot for plotting the image
    img_ transparency: between [0, 1] , if 0 image is not displayed
    but coordinates fit as if the image was in the background.

    return: image as background on the matplotlib axis
    '''
    img = mpimg.imread(img_loc)
    imgplot = ax.imshow(img, extent=extent, alpha=img_transparancy)
    return imgplot

#==============================================================================
#
#==============================================================================


def transform_variables_to_uniform_marginals(fish_file, VarA, VarB,
                                             data_source_str,
                                             out_plots_dir):
    '''
        transform variables to uniform marginals for Copulas
        this is used to plot the pure dependence between two
        variables
    '''
    if data_source_str == 'original_data':
        df_fish = calculate_fish_velocity(fish_file, wgs82, utm32)
    if data_source_str == 'filtered_data':
        df_fish = pd.read_csv(fish_file, index_col=0, parse_dates=True)
    varx = df_fish[VarA].values
    vary = df_fish[VarB].values
    xvals, yvals = np.squeeze(varx), np.squeeze(vary)

    ranks_R1i = (len(xvals) - stats.rankdata(xvals) + 0.5) / len(xvals)
    ranks_S1i = (len(yvals) - stats.rankdata(yvals) + 0.5) / len(yvals)

    _, (ax1, ax2) = plt.subplots(
        1, 2, sharey=True, sharex=True)  # , projection='3d')
    ax1.scatter(ranks_S1i, ranks_R1i, c='b', marker='.', alpha=0.5, s=0.5)

    ax2.hist2d(ranks_S1i, ranks_R1i, bins=10, normed=True)
    ax1.set_xlabel('Normed Ranked %s' % VarA)
    ax2.set_xlabel('Normed Ranked %s' % VarA)
    ax1.set_ylabel('Normed Ranked %s' % VarB)
    savefig('%s_marignals_var_%s_var_%s'
            % (data_source_str, VarA, VarB),
            out_plots_dir)
    return

#==============================================================================
#
#==============================================================================


def plot_histogram(fish_file, fish_nbr, variable_to_plt,
                   out_plots_dir, data_source_str,
                   var_thr=None, use_log=False):
    ''' fct to plot histogram for a variable'''
    if data_source_str == 'original_data':
        df_fish_0 = readDf(fish_file)
        df_fish = calculate_fish_velocity(df_fish_0, wgs82, utm32)
    if data_source_str == 'filtered_data':
        df_fish = pd.read_csv(fish_file, index_col=0, parse_dates=True)
    _, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=100)
    vals = df_fish[variable_to_plt].values
    if var_thr is not None:
        vals = vals[vals <= var_thr]
    if use_log:
        vals = np.log(vals)
    bins = np.arange(vals.min(), vals.max(), 0.5)
    center = (bins[:-1] + bins[1:]) / 2
    hist, _ = np.histogram(vals, bins=bins, density=False)
    ax.bar(center, hist, width=0.25, alpha=0.5,  # align='center',
           linewidth=0.51, color='blue', edgecolor='darkblue',
           label='%s Distribtuion' % variable_to_plt)
    ax.set_yticks(np.arange(0, hist.max(), vals.shape[0] / 20))
    ax.set_ylim([0, hist.max()])
    ax.set_title('Fish %s Distribution of %s with upper threshold of %s'
                 % (fish_nbr, variable_to_plt, str(var_thr)),
                 fontsize=fontsize)
    ax.set_xlabel('Classes', fontsize=fontsize)
    ax.set_ylabel('Frequency of Occurrence', fontsize=fontsize)
    savefig('%s_%s_distribution_%s'
            % (data_source_str, variable_to_plt, fish_nbr),
            out_plots_dir)
    plt.clf()
    plt.close()
    return

#==============================================================================
#
#==============================================================================


def plot_Var_values(img_loc, orig_station_file, fish_file,
                    fish_type_nbr, var_to_plt, var_bounds,
                    out_plots_dir, data_source_str):
    ''' fct to plot any variable (Velocity, HPe, RMSE) based on input file '''
    in_orig_stn_df = read_OrigStn_DF(orig_station_file)

    if data_source_str == 'original_data':
        df_fish_0 = readDf(fish_file)
        df_fish = calculate_fish_velocity(df_fish_0, wgs82, utm32)
    if data_source_str == 'filtered_data':
        df_fish = pd.read_csv(fish_file, index_col=0, parse_dates=True)

    print(df_fish)
    norm = mcolors.BoundaryNorm(boundaries=var_bounds, ncolors=256)
    fig, ax = plt.subplots(1, 1, figsize=(16, 12), dpi=100)

    plot_img(img_loc, ax)
    pcm = ax.scatter(df_fish['Longitude'].values,
                     df_fish['Latitude'].values,
                     c=df_fish[var_to_plt].values, s=0.05, alpha=0.5,
                     marker=',', cmap='RdBu_r', norm=norm,
                     vmin=df_fish[var_to_plt].values.min(),
                     vmax=df_fish[var_to_plt].values.max())
    ax.scatter(in_orig_stn_df['Longitude'].values,
               in_orig_stn_df['Latitude'].values,
               s=0.08, marker='x', c='k',
               label='Ref Stations', alpha=0.75)

    ax.set_xlim(extent[0], extent[1]), ax.set_ylim(extent[2], extent[3])

    cb = fig.colorbar(pcm, ax=ax, extend='max', orientation='vertical')
    cb.ax.tick_params(labelsize=labelsize)
    cb.ax.set_ylabel('%s values' % var_to_plt, fontsize=fontsize)
    cb.set_alpha(1), cb.draw_all()
    ax.set_xlabel('Longitude', fontsize=fontsize)
    ax.set_ylabel('Latitude', fontsize=fontsize)
    ax.tick_params(axis='x', labelsize=labelsize)
    ax.tick_params(axis='y', labelsize=labelsize)
    ax.set_title('%s Values for Fish: %s' % (var_to_plt, fish_type_nbr),
                 fontsize=fontsize, y=0.99)
    plt.legend(loc=0, fontsize=fontsize)
    plt.grid(alpha=0.5)
    savefig('%s_%s_%s' % (data_source_str, var_to_plt, fish_type_nbr),
            out_plots_dir)
    plt.clf()
    plt.close()
    return

#==============================================================================
#
#==============================================================================


def plot_variables(data_loc,
                   data_kmz_dir_4_names,
                   out_save_dir_fn,
                   data_source_str,
                   Velocity=False, RMSE=False, HPE=False):
    ''' fct used to plot scatter plot of all data files in given directory'''
    in_fish_files_dict = getFiles(data_loc, '.csv', data_kmz_dir_4_names)
    velocity_bounds = np.array([0, 0.25, 0.5, 1, 1.5, 2.])
    HPE_bounds = np.array([0, 3, 5, 10, 15, 20])
    RMSE_bounds = np.array([0, .01, .03, .05, .07, .1, 0.25, 1.0])
    for fish_type in in_fish_files_dict.keys():
        print('fish type is:', fish_type)
        for fish_file in in_fish_files_dict[fish_type]:
            print('fish file is', fish_file)
            if data_source_str == 'original_data':
                fish_tag_nbr = fish_type + '_' + fish_file[-9:-4]
            if data_source_str == 'filtered_data':
                fish_tag_nbr = fish_type + '_' + fish_file[-10:-5]
            if Velocity:
                plot_Var_values(img_loc, orig_station_file, fish_file,
                                fish_tag_nbr,
                                'Fish_Swimming_Velocity_in_m_per_s',
                                velocity_bounds, out_save_dir_fn,
                                data_source_str)
            if RMSE:
                plot_Var_values(img_loc, orig_station_file, fish_file,
                                fish_tag_nbr,
                                'RMSE', RMSE_bounds, out_save_dir_fn,
                                data_source_str)
            if HPE:
                plot_Var_values(img_loc, orig_station_file, fish_file,
                                fish_tag_nbr,
                                'HPE', HPE_bounds, out_save_dir_fn,
                                data_source_str)
            print('done saving data for: ', fish_file)

#==============================================================================
#
#==============================================================================


if __name__ == '__main__':

    # -----------------------------------------------------------------------
    # call function here
    if plot_vel_hpe_rmse_original_data:

        plot_variables(orig_data_dir,
                       dir_kmz_for_fish_names,
                       out_save_dir,
                       'original_data',
                       Velocity=True, RMSE=True, HPE=True)

    if plot_vel_hpe_rmse_filtered_data:
        plot_variables(filtered_out_data,
                       dir_kmz_for_fish_names,
                       out_save_dir,
                       'filtered_data',
                       Velocity=True, RMSE=False, HPE=False)
    # -----------------------------------------------------------------------
    if plot_original_histograms_and_marginals_vel_hpe_rmse_:
        # get files for every fish type
        in_orig_fish_files_dict = getFiles(orig_data_dir, '.csv',
                                           dir_kmz_for_fish_names)

        for fish_type in in_orig_fish_files_dict.keys():
            for fish_file in in_orig_fish_files_dict[fish_type]:
                print(fish_file)

                fish_nbr = fish_type + '_' + fish_file[-9:-4]
                try:
                    transform_variables_to_uniform_marginals(fish_file,
                                                             'HPE',
                                                             'RMSE',
                                                             'original_data',
                                                             out_save_dir)

                    plot_histogram(fish_file, fish_nbr, 'HPE',  # 'RMSE'
                                   out_save_dir, 'original_data',
                                   var_thr=30, use_log=False)
                except Exception as msg:
                    print(msg)
                break
            break

    if plot_filtered_histograms_and_marginals_vel_hpe_rmse_:
        # get files for every fish type
        in_filtered_fish_files_dict = getFiles(filtered_out_data, '.csv',
                                               dir_kmz_for_fish_names)

        for fish_type in in_filtered_fish_files_dict.keys():
            for fish_file in in_filtered_fish_files_dict[fish_type]:
                print(fish_file)

                fish_nbr = fish_type + '_' + fish_file[-10:-5]
                try:
                    transform_variables_to_uniform_marginals(fish_file,
                                                             'HPE',
                                                             'RMSE',
                                                             'filtered_data',
                                                             out_save_dir)

                    plot_histogram(fish_file, fish_nbr, 'HPE',  # 'RMSE'
                                   out_save_dir, 'filtered_data',
                                   var_thr=10, use_log=False)
                except Exception as msg:
                    print(msg)
                break
            break
