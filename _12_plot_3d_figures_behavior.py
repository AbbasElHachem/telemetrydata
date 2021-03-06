#!/usr/bin/env python
# coding: utf-8

"""
Created on 25.03.2019

@author: EL Hachem Abbas,
Institut fuer Wasser- und Umweltsystemmodellierung - IWS

"""
# %%

from __future__ import division


import os
import time
import timeit


import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapefile
import datetime

from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator
# from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rc
from matplotlib import rcParams
from datetime import timedelta
from astral import Astral, Location


from _00_define_main_directories import (dir_kmz_for_fish_names,
                                         out_data_dir,
                                         fish_shp_path,
                                         river_shp_path)

from _01_filter_fish_points_keep_only_in_river import getFiles
from _02_filter_fish_data_based_on_HPE_Vel_RMSE import convert_coords_fr_wgs84_to_utm32_
from _02_filter_fish_data_based_on_HPE_Vel_RMSE import wgs82, utm32

plt.ioff()
# get_ipython().run_line_magic('matplotlib', 'inline')

rc('font', size=16)
rc('font', family='serif')
rc('axes', labelsize=20)
rcParams['axes.labelpad'] = 35

astral = Astral()

#==============================================================================
# Def project location
#==============================================================================
project_location = Location()

city_name = 'Altusried'
longitude = 47.80997
latitude = 10.22982
elevation = 654

project_location.name = city_name
project_location.region = 'Germany'
project_location.latitude = latitude
project_location.longitude = longitude
project_location.timezone = 'Europe/Berlin'
project_location.elevation = elevation
project_location.sun()
#==============================================================================
#
#==============================================================================
# %%


# def find_day_dawn_dusk_info(fish_file):
#     df_fish = pd.read_csv(fish_file,
#                           sep=',',
#                           index_col=0,
#                           parse_dates=True,
#                           engine='c',
#                           infer_datetime_format=True)
#
#     for fish_obsv_time in df_fish.index:
#         sun = project_location.sun(date=fish_obsv_time,
#                                    local=True)
#
#         dawn = sun['sunrise'] - pd.Timedelta(hours=1)
#
#         day_start, day_end = sun['sunrise'], sun['sunset']
#         dusk = sun['sunset'] + pd.Timedelta(hours=1)
#         night_start, night_end = dusk, dawn
#
#         # print('Dawn:    %s' % str(dawn))
#         # print('Day Start: %s' % str(day_start))
#         # print('Day End: %s' % str(day_end))
#         # print('Dusk: %s' % str(dusk))
#         # print('Night Start: %s' % str(night_start))
#         # print('Night End: %s' % str(night_end))
#
#         # define colors for periods 1: dawn, 2: day, 3: dusk, 4:night
#         try:
#             if dawn.replace(tzinfo=None) <= fish_obsv_time < day_start.replace(tzinfo=None):
#                 df_fish.loc[fish_obsv_time, 'color_3d_plot'] = 'g'  # 'Dawn'
#
#             elif day_start.replace(tzinfo=None) <= fish_obsv_time < day_end.replace(tzinfo=None):
#                 df_fish.loc[fish_obsv_time, 'color_3d_plot'] = 'gold'  # 'Day'
#
#             elif day_end.replace(tzinfo=None) <= fish_obsv_time < dusk.replace(tzinfo=None):
#                 df_fish.loc[fish_obsv_time,
#                             'color_3d_plot'] = 'darkred'  # 'Dusk'
#
#         # elif dusk.replace(tzinfo=None) <= fish_obsv_time < (dawn.replace(tzinfo=None) +
#         #                                                    pd.Timedelta(days=1)):  # add one day because next day in the morning
# #             df_fish.loc[fish_obsv_time,
# #                         'color_3d_plot'] = 'darkblue'  # 'Night'
#             else:
#                 df_fish.loc[fish_obsv_time,
#                             'color_3d_plot'] = 'darkblue'
#             #print('Time of observation not found:', fish_obsv_time)
# #             raise Exception
#         except Exception as msg:
#             print(fish_obsv_time, 'Error', msg)
#         print('Observation Time', fish_obsv_time, ';Time period:',
#               df_fish.loc[fish_obsv_time, 'color_3d_plot'])
#
#         colors_for_plot = df_fish['color_3d_plot'].values.ravel()
#     return colors_for_plot
#==============================================================================
#
#==============================================================================


def find_day_dawn_dusk_info_per_idx(time_idx):

    sun = project_location.sun(date=time_idx,
                               local=True)

    dawn = sun['sunrise'] - pd.Timedelta(hours=1)

    day_start, day_end = sun['sunrise'], sun['sunset']
    dusk = sun['sunset'] + pd.Timedelta(hours=1)
    #night_start, night_end = dusk, dawn

    if dawn.replace(tzinfo=None) <= time_idx < day_start.replace(tzinfo=None):
        return 'g'  # 'Dawn'

    elif day_start.replace(tzinfo=None) <= time_idx < day_end.replace(tzinfo=None):
        return 'gold'  # 'Day'

    elif day_end.replace(tzinfo=None) <= time_idx < dusk.replace(tzinfo=None):
        return 'darkred'  # 'Dusk'

    else:
        return 'darkblue'  # 'Night'
    #print('Time of observation not found:', fish_obsv_time)

#==============================================================================
#
#==============================================================================


def hour_rounder(t):
    # Rounds to nearest hour by adding a timedelta hour if minute >= 30
    return (t.replace(second=0, microsecond=0, minute=0, hour=t.hour)
            + timedelta(hours=t.minute // 30))
#==============================================================================
#
#==============================================================================


def plot_3d_plot_tiomeofday_as_colr(fish_file, fish_nbr, flow_cat,
                                    fishshp, rivershp, out_save_dir):

    fig = plt.figure(figsize=(40, 20), dpi=100)
    # fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    # ax = fig.gca(projection='3d')
    ax = fig.add_subplot(111, projection='3d')
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.set_major_locator(LinearLocator(10))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.04f'))
    ax.yaxis.set_major_locator(LinearLocator(10))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.04f'))

    # get colors for plot
    # colors_3d_plot = find_day_dawn_dusk_info(fish_file)
#      = in_df.apply(lambda x: fxy(x['A'], x['B']), axis=1)
    print('plotting')
    in_df = pd.read_csv(fish_file, index_col=0, parse_dates=True)
    x_vals = in_df.Longitude.values
    y_vals = in_df.Latitude.values
    in_df['Time'] = in_df.index
    in_df['colors_3d_plot'] = in_df.Time.apply(
        lambda x: find_day_dawn_dusk_info_per_idx(x))

    colors_3d_plot = in_df['colors_3d_plot'].values
    z_vals = in_df.index.to_pydatetime()
    dates_formatted = [pd.to_datetime(d) for d in z_vals]
    z_vals_ix = np.arange(0, len(z_vals), 1)

#     bounds = {0: [0, 4], 1: [4, 10], 2: [10, 16], 3: [16, 22], 4: [22, 0]}

#     for ix, val in zip(in_df.index, in_df.index.hour):
#        for k, v in bounds.items():
#             if v[0] <= val <= v[1]:
#                 in_df.loc[ix, 'colors'] = clrs[k]
#         if val < bounds[0][0]:
#             in_df.loc[ix, 'colors'] = clrs[0]
#         if val > bounds[4][0]:
#             in_df.loc[ix, 'colors'] = clrs[4]

    if 'g' in in_df['colors_3d_plot'].values:
        # find minimal dusk / dawn start and end
        min_dawn_start = in_df[in_df['colors_3d_plot']
                               == 'g'].index.min().round('60min')
        max_dawn_end = in_df[in_df['colors_3d_plot']
                             == 'g'].index.max().round('60min')
        if min_dawn_start.hour >= max_dawn_end.hour:
            max_dawn_end += pd.Timedelta(hours=1)
    if 'gold' in in_df['colors_3d_plot'].values:
        min_day_start = in_df[in_df['colors_3d_plot']
                              == 'gold'].index.min().round('60min')
        max_day_end = in_df[in_df['colors_3d_plot']
                            == 'gold'].index.max().round('60min')
        if min_day_start.hour >= max_day_end.hour:
            max_day_end += pd.Timedelta(hours=1)
    if 'darkred' in in_df['colors_3d_plot'].values:
        min_dusk_start = in_df[in_df['colors_3d_plot']
                               == 'darkred'].index.min().round('60min')
        max_dusk_end = in_df[in_df['colors_3d_plot']
                             == 'darkred'].index.max().round('60min')
        if min_dusk_start.hour >= max_dusk_end.hour:
            max_dusk_end += pd.Timedelta(hours=1)
    if 'darkblue' in in_df['colors_3d_plot'].values:
        min_night_start = in_df[in_df['colors_3d_plot']
                                == 'darkblue'].index.min().round('60min')
        max_night_end = in_df[in_df['colors_3d_plot']
                              == 'darkblue'].index.max().round('60min')

        if min_night_start.hour >= max_night_end.hour:
            max_night_end += pd.Timedelta(hours=1)
    #==========================================================================
    # FIND WHICH PERIOD (15 possible combination)
    #==========================================================================
    # if only dawn available (1)
    if ('g' in in_df['colors_3d_plot'].values) and (
            'gold' not in in_df['colors_3d_plot'].values) and (
                'darkred' not in in_df['colors_3d_plot'].values) and (
                'darkblue' not in in_df['colors_3d_plot'].values):
        ticks = [min(min_dawn_start.hour, max_dawn_end.hour),
                 max(min_dawn_start.hour, max_dawn_end.hour)]

        ticks_str = ['Dawn']
        clrs = ['g']
    # all four periods are available (1 ,2, 3, 4)
    if ('g' in in_df['colors_3d_plot'].values) and (
            'darkred' in in_df['colors_3d_plot'].values) and (
            'gold' in in_df['colors_3d_plot'].values) and (
            'darkblue' in in_df['colors_3d_plot'].values):

        if max_dusk_end.hour <= min_dusk_start.hour:
            max_dusk_end += pd.Timedelta(hours=1)
        if max_dawn_end.hour <= min_dawn_start.hour:
            max_dawn_end += pd.Timedelta(hours=1)
        ticks = [0, min_dawn_start.hour,
                 max_dawn_end.hour,
                 min_dusk_start.hour,
                 max_dusk_end.hour, 24]

        ticks_str = ['Night', 'Dawn', 'Day', 'Dusk', 'Night']
        clrs = ['darkblue', 'green', 'gold', 'darkred', 'darkblue']
    # if only dawn, day, dusk available (1, 2, 3)
    if ('g' in in_df['colors_3d_plot'].values) and (
            'gold' in in_df['colors_3d_plot'].values) and (
                'darkred' in in_df['colors_3d_plot'].values) and (
                'darkblue' not in in_df['colors_3d_plot'].values):
        ticks = [min_dawn_start.hour, max_dawn_end.hour,
                 min_day_start.hour, max_day_end.hour,
                 min_dusk_start.hour, max_dusk_end.hour]

        ticks_str = ['Dawn', 'Day', 'Dusk']
        clrs = ['green', 'gold', 'darkred']
    # if only dawn, day, available (1, 2)
    if ('g' in in_df['colors_3d_plot'].values) and (
            'gold' in in_df['colors_3d_plot'].values) and (
                'darkred' not in in_df['colors_3d_plot'].values) and (
                'darkblue' not in in_df['colors_3d_plot'].values):
        if max_day_end.hour == min_day_start.hour:
            max_day_end += pd.Timedelta(hours=1)
        ticks = [min_dawn_start.hour, max_dawn_end.hour,
                 max_day_end.hour]

        ticks_str = ['Dawn', 'Day']
        clrs = ['green', 'gold']
    # if only dawn, dusk, available (1, 3)
    if ('g' in in_df['colors_3d_plot'].values) and (
            'gold' not in in_df['colors_3d_plot'].values) and (
                'darkred' in in_df['colors_3d_plot'].values) and (
                'darkblue' not in in_df['colors_3d_plot'].values):
        ticks = [min_dawn_start.hour, max_dawn_end.hour,
                 min_dusk_start.hour, max_dusk_end.hour]

        ticks_str = ['Dawn', 'Dusk']
        clrs = ['green',  'darkred']
    # in only Dawn, Dusk, night are available (1 , 3, 4)
    if ('g' in in_df['colors_3d_plot'].values) and (
            'gold' not in in_df['colors_3d_plot'].values) and (
            'darkred' in in_df['colors_3d_plot'].values) and (
            'darkblue' in in_df['colors_3d_plot'].values):
        ticks = [0, min_dawn_start.hour,  max_dawn_end.hour,
                 min_dusk_start.hour, max_dusk_end.hour,
                 min_night_start, 24]
        ticks_str = ['Night', 'Dawn', 'Dusk', 'Night']
        clrs = ['darkblue', 'green', 'darkred', 'darkblue']
    # in only Dawn, Day, night are available (1 , 2, 4)
    if ('g' in in_df['colors_3d_plot'].values) and (
            'gold' in in_df['colors_3d_plot'].values) and (
                'darkred' not in in_df['colors_3d_plot'].values) and (
            'darkblue' in in_df['colors_3d_plot'].values):
        #         if max_dawn_end.hour == min_day_start.hour:
        #             min_day_start += pd.Timedelta(hours=1)
        ticks = [0, min_dawn_start.hour,  max_dawn_end.hour,
                 max_day_end.hour, 24]
        ticks_str = ['Night', 'Dawn', 'Day', 'Night']
        clrs = ['darkblue', 'green', 'gold', 'darkblue']
    # in only Dawn,  night are available (1 , 4)
    if ('g' in in_df['colors_3d_plot'].values) and (
            'gold' not in in_df['colors_3d_plot'].values) and (
                'darkred' not in in_df['colors_3d_plot'].values) and (
            'darkblue' in in_df['colors_3d_plot'].values):
        ticks = [0, min_dawn_start.hour,  max_dawn_end.hour, 24]
        ticks_str = ['Night', 'Dawn', 'Night']
        clrs = ['darkblue', 'green', 'darkblue']
    # if only day available (2)
    if ('g' not in in_df['colors_3d_plot'].values) and (
            'gold' in in_df['colors_3d_plot'].values) and (
                'darkred' not in in_df['colors_3d_plot'].values) and (
                'darkblue' not in in_df['colors_3d_plot'].values):
        ticks = [min(min_day_start.hour, max_day_end.hour),
                 max(min_day_start.hour, max_day_end.hour)]

        ticks_str = ['Day']
        clrs = ['gold']
    # if only day and dusk available (2, 3)
    if ('g' not in in_df['colors_3d_plot'].values) and (
            'gold' in in_df['colors_3d_plot'].values) and (
                'darkred' in in_df['colors_3d_plot'].values) and (
                'darkblue' not in in_df['colors_3d_plot'].values):

        ticks = [min(min_day_start.hour, max_day_end.hour),
                 min(min_dusk_start.hour, max_dusk_end.hour),
                 max(min_dusk_start.hour, max_dusk_end.hour)]

        ticks_str = ['Day', 'Dusk']
        clrs = ['gold', 'darkred']
    # in only Day, dusk, night are available (2,3, 4)
    if ('g' not in in_df['colors_3d_plot'].values) and (
            'gold' in in_df['colors_3d_plot'].values) and (
                'darkred' in in_df['colors_3d_plot'].values) and (
            'darkblue' in in_df['colors_3d_plot'].values):
        ticks = [0, min_day_start.hour, max_day_end.hour,
                 min_dusk_start.hour, max_dusk_end.hour,
                 min_night_start, 24]
        ticks_str = ['Night', 'Day', 'Dusk', 'Night']
        clrs = ['darkblue', 'gold', 'darkred', 'darkblue']
    # in only Day, night are available (2, 4)
    if ('g' not in in_df['colors_3d_plot'].values) and (
            'gold' in in_df['colors_3d_plot'].values) and (
                'darkred' not in in_df['colors_3d_plot'].values) and (
            'darkblue' in in_df['colors_3d_plot'].values):
        if min_day_start.hour < max_day_end.hour:
            ticks = [0, min_day_start.hour, max_day_end.hour, 24]
        else:
            ticks = [0, max_day_end.hour, min_day_start.hour, 24]
        ticks_str = ['Night', 'Day', 'Night']
        clrs = ['darkblue', 'gold', 'darkblue']
    # if only dusk available (3)
    if ('g' not in in_df['colors_3d_plot'].values) and (
            'gold' not in in_df['colors_3d_plot'].values) and (
                'darkred' in in_df['colors_3d_plot'].values) and (
                'darkblue' not in in_df['colors_3d_plot'].values):
        ticks = [min(min_dusk_start.hour, max_dusk_end.hour),
                 max(min_dusk_start.hour, max_dusk_end.hour)]

        ticks_str = ['Dusk']
        clrs = ['darkred']

    # if only dusk, night available (3, 4)
    if ('g' not in in_df['colors_3d_plot'].values) and (
            'gold' not in in_df['colors_3d_plot'].values) and (
                'darkred' in in_df['colors_3d_plot'].values) and (
                'darkblue' in in_df['colors_3d_plot'].values):
        ticks = [0, min(min_dusk_start.hour, max_dusk_end.hour),
                 max(min_dusk_start.hour, max_dusk_end.hour),
                 24]

        ticks_str = ['Night', 'Dusk', 'Night']
        clrs = ['darkblue', 'darkred', 'darkblue']
    # if only night available ( 4)
    if ('g' not in in_df['colors_3d_plot'].values) and (
            'gold' not in in_df['colors_3d_plot'].values) and (
                'darkred' not in in_df['colors_3d_plot'].values) and (
                'darkblue' in in_df['colors_3d_plot'].values):
        ticks = [0, 24]

        ticks_str = ['Night', 'Night']
        clrs = ['darkblue', 'darkblue']
    #==========================================================================
    # position factor
    #==========================================================================
    if len(ticks_str) == 5:
        div_pos_fact = 10.0
    elif len(ticks_str) == 4:
        div_pos_fact = 8
    elif len(ticks_str) == 3:
        div_pos_fact = 6
    elif len(ticks_str) == 2:
        div_pos_fact = 5
    else:
        div_pos_fact = 2

    ax.scatter3D(x_vals, y_vals, zs=z_vals_ix, zdir='z',
                 c=colors_3d_plot, alpha=0.65,
                 marker=',', s=8)

    ax.plot3D(x_vals, y_vals, zs=z_vals_ix, zdir='z',
              c='k', alpha=0.25, linewidth=0.75)

    sf_river = shapefile.Reader(rivershp)
    for shape_ in sf_river.shapeRecords():
        x0 = [i[0] for i in shape_.shape.points[:][::-1]]
        y0 = [i[1] for i in shape_.shape.points[:][::-1]]
        lon, lat = convert_coords_fr_wgs84_to_utm32_(utm32, wgs82, x0, y0)
        ax.plot3D(lon, lat, 0, zdir='z',
                  color='k', alpha=0.25,
                  marker='.', linewidth=0.51,
                  label='River Boundary Area')

    sf = shapefile.Reader(fishshp)

    for shape_ in sf.shapeRecords():
        x0 = [i[0] for i in shape_.shape.points[:][::-1]]
        y0 = [i[1] for i in shape_.shape.points[:][::-1]]
        ax.plot3D(x0, y0, 0, zdir='z',
                  color='k', alpha=0.65,
                  marker='+', linewidth=1,
                  label='Fish Pass Area')

    ax.scatter3D(10.2247927, 47.8186509, zs=z_vals_ix, zdir='z',
                 c='maroon', alpha=0.5, marker='D', s=40,
                 label='Fish pass entrance')
    ax.zaxis.set_ticks(
        z_vals_ix[::int(np.round(z_vals_ix.shape[0] / 15))])
    ax.zaxis.set_ticklabels(
        dates_formatted[::int(
            np.round(z_vals_ix.shape[0] / 15))])
    ax.set_xlabel('Longitude (x-axis)')
    ax.set_ylabel('Latitude (y-axis)')

    ax.set_title('Fish_%s_Flow_%s_colors_refer_to_%s'
                 % (fish_nbr, flow_cat,
                    'Time_of_day_h'))

    ax.set_xlim(min(lon), max(lon)), ax.set_ylim(min(lat), max(lat))

    #_, idx = np.unique(in_df['colors_3d_plot'].values, return_index=True)

    #clrs = in_df['colors_3d_plot'].values[np.sort(idx)]

    # night, dawn, day, dusk
    print(clrs)
    cmap = mcolors.ListedColormap(clrs)

    norm = mcolors.BoundaryNorm(ticks, cmap.N)
    ax_legend = fig.add_axes([0.1725, 0.07525, 0.68, 0.0225], zorder=3)
    cb = mpl.colorbar.ColorbarBase(ax_legend, ticks=ticks,
                                   boundaries=ticks, norm=norm, cmap=cmap,
                                   orientation='horizontal')
    cb.set_label('Time_of_day_h', rotation=0)
    ax.set_aspect('auto')

    cb.ax.get_xaxis().set_ticks([])
    print(ticks_str)
    for j, lab in enumerate(ticks_str):
        cb.ax.text((2 * j + 1) / div_pos_fact, .5,
                   lab, ha='center', va='center',
                   color='w', fontweight='bold')
    # cb.ax.set_xticklabels(ticks_str)

    ax.view_init(25, 275)

    ax.legend(loc=0)
    cb.draw_all()
    cb.set_alpha(1)
    plt.savefig(os.path.join(out_save_dir,
                             '3d_%s_%s_%s.png'
                             % (fish_nbr, flow_cat,
                                'Period_of_day_')))
    plt.close()
    print('done plotting')
    return

# %%

#==============================================================================
#
#==============================================================================


def plot_3d_plot_flow_as_color(fish_file, fish_nbr, flow_cat,
                               fishshp, rivershp, out_save_dir):

    fig = plt.figure(figsize=(40, 20), dpi=75)
#     # fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
#     # ax = fig.gca(projection='3d')
    ax = fig.add_subplot(111, projection='3d')
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.set_major_locator(LinearLocator(10))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.04f'))
    ax.yaxis.set_major_locator(LinearLocator(10))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.04f'))

    in_df = pd.read_csv(fish_file, index_col=0, parse_dates=True)
    x_vals = in_df.Longitude.values
    y_vals = in_df.Latitude.values

    z_vals = in_df.index.to_pydatetime()
    dates_formatted = [pd.to_datetime(d) for d in z_vals]
    z_vals_ix = np.arange(0, len(z_vals), 1)

    z_vals_colrs = in_df['velM_%s' % flow_cat[-2:]].values

    bounds = {0: [0.0, 0.1], 1: [0.1, 0.2], 2: [0.2, 0.4],
              3: [0.4, 0.6], 4: [0.6, 0.8]}
    clrs = ['navy', 'blue', 'g', 'gold', 'darkorange', 'red']
    cmap = mcolors.ListedColormap(clrs)
    for ix, val in zip(in_df.index, z_vals_colrs):
        for k, v in bounds.items():
            if v[0] <= val <= v[1]:
                in_df.loc[ix, 'colors'] = clrs[k]
        if val < bounds[0][0]:
            in_df.loc[ix, 'colors'] = clrs[0]
        if val > bounds[4][1]:
            in_df.loc[ix, 'colors'] = clrs[4]

    colors_ = in_df.colors
    ticks = [0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.81]

    ax.scatter3D(x_vals, y_vals, zs=z_vals_ix, zdir='z',
                 c=colors_, alpha=0.5, marker=',', s=8)

    ax.plot3D(x_vals, y_vals, zs=z_vals_ix, zdir='z',
              c='k', alpha=0.25, linewidth=0.55)

    sf_river = shapefile.Reader(rivershp)

    for shape_ in sf_river.shapeRecords():
        x0 = [i[0] for i in shape_.shape.points[:][::-1]]
        y0 = [i[1] for i in shape_.shape.points[:][::-1]]
        lon, lat = convert_coords_fr_wgs84_to_utm32_(utm32, wgs82, x0, y0)
        ax.plot3D(lon, lat, 0, zdir='z',
                  color='k', alpha=0.25,
                  marker='.', linewidth=0.51,
                  label='River Boundary Area')

    sf = shapefile.Reader(fishshp)
    for shape_ in sf.shapeRecords():
        x0 = [i[0] for i in shape_.shape.points[:][::-1]]
        y0 = [i[1] for i in shape_.shape.points[:][::-1]]
        ax.plot3D(x0, y0, 0, zdir='z',
                  color='k', alpha=0.65,
                  marker='+', linewidth=1,
                  label='Fish Pass Area')

    ax.scatter3D(10.2247927, 47.8186509, zs=z_vals_ix, zdir='z',
                 c='maroon', alpha=0.5, marker='D', s=40,
                 label='Fish pass entrance')
    ax.zaxis.set_ticks(
        z_vals_ix[::int(np.round(z_vals_ix.shape[0] / 15))])
    ax.zaxis.set_ticklabels(
        dates_formatted[::int(np.round(z_vals_ix.shape[0] / 15))])
    ax.set_xlabel('Longitude (x-axis)')
    ax.set_ylabel('Latitude (y-axis)')

    ax.set_title('Fish_%s_Flow_%s_colors_refer_to_%s'
                 % (fish_nbr, flow_cat,
                    'Flow_velocity_m_per_s'), y=0.98)
#     ax.set_xlim(10.223, 10.226), ax.set_ylim(47.818, 47.820)
    ax.set_xlim(min(lon), max(lon)), ax.set_ylim(min(lat), max(lat))
    norm = mcolors.BoundaryNorm(ticks, cmap.N)
    ax_legend = fig.add_axes([0.1725, 0.07525, 0.68, 0.0225], zorder=3)
    cb = mpl.colorbar.ColorbarBase(ax_legend, ticks=ticks, extend='max',
                                   boundaries=ticks, norm=norm, cmap=cmap,
                                   orientation='horizontal')

    cb.set_label('Flow_velocity_m_per_s')
#     ax.set_aspect('auto')

#     for angle in range(0, 360):
    ax.view_init(25, 275)

    ax.legend(loc='upper right', frameon=True)
    cb.draw_all()
    cb.set_alpha(1)
#     plt.tight_layout()
    plt.savefig(os.path.join(out_save_dir,
                             '3d_%s_%s_%s_.png'
                             % (fish_nbr, flow_cat,
                                'Flow_velocity_m_per_s')))
    plt.close()
    return


#==============================================================================
#
#==============================================================================


def plot_3d_plot_group_as_color(fish_file, fish_nbr, flow_cat,
                                fishshp, rivershp, out_save_dir):

    fig = plt.figure(figsize=(40, 20), dpi=75)
#     # fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
#     # ax = fig.gca(projection='3d')
    ax = fig.add_subplot(111, projection='3d')
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.set_major_locator(LinearLocator(10))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.04f'))
    ax.yaxis.set_major_locator(LinearLocator(10))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.04f'))

    in_df = pd.read_feather(fish_file, use_threads=4)
    in_df.set_index('Time', inplace=True)
    in_df.index = pd.to_datetime(in_df.index, format='%Y-%m-%d %H:%M:%S.%f')
    x_vals = in_df.Longitude.values
    y_vals = in_df.Latitude.values

    z_vals = in_df.index.to_pydatetime()
    dates_formatted = [pd.to_datetime(d) for d in z_vals]
    z_vals_ix = np.arange(0, len(z_vals), 1)

    z_vals_colrs = in_df['group'].values

    bounds = {0: 0, 1: 1, 2: 2}
    clrs = ['navy', 'darkorange', 'darkred']
    cmap = mcolors.ListedColormap(clrs)
    for ix, val in zip(in_df.index, z_vals_colrs):
        for k, v in bounds.items():
            if v == val:
                in_df.loc[ix, 'colors'] = clrs[k]

    colors_ = in_df.colors
    ticks = [0, 1, 2]

    ax.scatter3D(x_vals, y_vals, zs=z_vals_ix, zdir='z',
                 c=colors_, alpha=0.25, marker=',', s=8)

    ax.plot3D(x_vals, y_vals, zs=z_vals_ix, zdir='z',
              c='k', alpha=0.25, linewidth=0.55)

    sf_river = shapefile.Reader(rivershp)

    for shape_ in sf_river.shapeRecords():
        x0 = [i[0] for i in shape_.shape.points[:][::-1]]
        y0 = [i[1] for i in shape_.shape.points[:][::-1]]
        lon, lat = convert_coords_fr_wgs84_to_utm32_(utm32, wgs82, x0, y0)
        ax.plot3D(lon, lat, 0, zdir='z',
                  color='k', alpha=0.25,
                  marker='.', linewidth=0.51,
                  label='River Boundary Area')

    sf = shapefile.Reader(fishshp)
    for shape_ in sf.shapeRecords():
        x0 = [i[0] for i in shape_.shape.points[:][::-1]]
        y0 = [i[1] for i in shape_.shape.points[:][::-1]]
        ax.plot3D(x0, y0, 0, zdir='z',
                  color='k', alpha=0.65,
                  marker='+', linewidth=1,
                  label='Fish Pass Area')

    ax.scatter3D(10.2247927, 47.8186509, zs=z_vals_ix, zdir='z',
                 c='maroon', alpha=0.5, marker='D', s=40,
                 label='Fish pass entrance')
    ax.zaxis.set_ticks(
        z_vals_ix[::int(np.round(z_vals_ix.shape[0] / 15))])
    ax.zaxis.set_ticklabels(
        dates_formatted[::int(np.round(z_vals_ix.shape[0] / 15))])
    ax.set_xlabel('Longitude (x-axis)')
    ax.set_ylabel('Latitude (y-axis)')

    ax.set_title('Fish_%s_Flow_%s_colors_refer_to_%s'
                 % (fish_nbr, flow_cat,
                    'Behaviour_group'), y=0.98)
#     ax.set_xlim(10.223, 10.226), ax.set_ylim(47.818, 47.820)
    ax.set_xlim(min(lon), max(lon)), ax.set_ylim(min(lat), max(lat))
    norm = mcolors.BoundaryNorm(ticks, cmap.N)
    ax_legend = fig.add_axes([0.1725, 0.07525, 0.68, 0.0225], zorder=3)
    cb = mpl.colorbar.ColorbarBase(ax_legend, ticks=ticks, extend='max',
                                   boundaries=ticks, norm=norm, cmap=cmap,
                                   orientation='horizontal')

    cb.set_label('Behaviour_group')
#     ax.set_aspect('auto')

#     for angle in range(0, 360):
    ax.view_init(25, 275)

    ax.legend(loc='upper right', frameon=True)
    cb.draw_all()
    cb.set_alpha(1)
#     plt.tight_layout()
    plt.savefig(os.path.join(out_save_dir,
                             '3d_%s_%s_%s_.png'
                             % (fish_nbr, flow_cat,
                                'Behaviour_group')))
    plt.close()
    return
# %%


if __name__ == '__main__':

    START = timeit.default_timer()
    print('Program started at: ', time.asctime())

#     in_fish_files_dict = getFiles(r'C:\Users\hachem\Desktop\Work_with_Matthias_Schneider\out_plots_abbas\Filtered_data',
# '.csv')
#     fish_file = r'C:\Users\hachem\Desktop\Work_with_Matthias_Schneider\out_plots_abbas\df_fish_flow_combined_with_angles\fish_barbel_46838_with_flow_data_10_and_angles_and_behaviour.ft'
#     fish_nbr = 'barbel_46838'
#     flow_cat = '10'

#     plot_3d_plot_group_as_color(fish_file,
#                                 fish_nbr,
#                                 flow_cat,
#                                 fish_shp_path,
#                                 river_shp_path,
#                                 out_data_dir)
    in_fish_files_dict = getFiles(
        r'C:\Users\hachem\Desktop\Work_with_Matthias_Schneider'
        r'\out_plots_abbas\df_fish_flow_combined_with_angles',
        '.csv', dir_kmz_for_fish_names)  #

    for fish_type in in_fish_files_dict.keys():
        for fish_file in in_fish_files_dict[fish_type][2:]:
            print(fish_file)

            fish_nbr = fish_type + fish_file[-47:-42]
#             # '_all_data_' + fish_file[-22:-17]  #    # fish_file[-32:-27]
            flow_cat = fish_file[-11:-5]
#             # 'not_considered'  # fish_file[-11:-5]
            print('fish number: ', fish_nbr, 'flow cat: ',  flow_cat)
            #raise Exception
#             if fish_nbr == '1_grayling46867' and flow_cat == 'cat_80':
#             if fish_nbr == '2_barbel46853':# and flow_cat == 'cat_80':
            # 2_barbel46854 cat_60
            try:
                plot_3d_plot_tiomeofday_as_colr(fish_file, fish_nbr, flow_cat,
                                                fish_shp_path, river_shp_path,
                                                out_data_dir)

    #                 plot_3d_plot_flow_as_color(fish_file, fish_nbr, flow_cat,
    #                                            fish_shp_path, river_shp_path,
    #                                            out_data_dir)
    #
            except Exception as msg:
                print(msg)
                continue
        #             break
#         break
    END = timeit.default_timer()

    print('Program ended at: ', time.asctime(),
          'Runtime was %0.4f seconds' % (END - START))
