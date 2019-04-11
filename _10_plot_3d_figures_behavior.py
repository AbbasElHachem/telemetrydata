#!/usr/bin/env python
# coding: utf-8

"""
Created on 25.03.2019

@author: EL Hachem Abbas,
Institut fuer Wasser- und Umweltsystemmodellierung - IWS

"""
#%%


from __future__ import division

from matplotlib.ticker import LinearLocator
# from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rc
from matplotlib import rcParams

from _02_filter_fish_data_based_on_HPE_Vel_RMSE import convert_coords_fr_wgs84_to_utm32_
from _02_filter_fish_data_based_on_HPE_Vel_RMSE import wgs82, utm32
import os
import time
import timeit

from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapefile
plt.ioff()
# get_ipython().run_line_magic('matplotlib', 'inline')

rc('font', size=16)
rc('font', family='serif')
rc('axes', labelsize=20)
rcParams['axes.labelpad'] = 35


#%%


def getFiles(data_dir_, file_ext_str, dir_kmz_for_fish_names):
    ''' function to get files based on dir and fish name'''

    def list_all_full_path(ext, file_dir):
        import fnmatch
        """
        Purpose: To return full path of files in all dirs of
        a given folder with a
        given extension in ascending order.
        Description of the arguments:
            ext (string) = Extension of the files to list \
                e.g. '.txt', '.tif'.
            file_dir (string) = Full path of the folder in which the files \
                reside.
        """
        new_list = []
        patt = '*' + ext
        for root, _, files in os.walk(file_dir):
            for elm in files:
                if fnmatch.fnmatch(elm, patt):
                    full_path = os.path.join(root, elm)
                    new_list.append(full_path)
        return(sorted(new_list))

    def get_file_names_per_fish_name(dir_fish_names_files):
        '''function to get all file names related to each fish '''
        fish_names = [name for name in os.listdir(dir_fish_names_files)
                      if os.path.isdir(os.path.join(dir_fish_names_files,
                                                    name))]
        dict_fish = {k: [] for k in fish_names}
        for ix, key_ in enumerate(dict_fish.keys()):
            files_per_fish = os.listdir(os.path.join(
                dir_fish_names_files, fish_names[ix]))
            dict_fish[key_] = files_per_fish
        return dict_fish

    dict_fish = get_file_names_per_fish_name(dir_kmz_for_fish_names)
    dict_files_per_fish = {k: [] for k in dict_fish.keys()}
    dfs_files = []
    for r, _dir, f in os.walk(data_dir_):
        for fs in f:
            if fs.endswith(file_ext_str):
                dfs_files.append(os.path.join(r, fs))
    assert len(dfs_files) > 0, 'Wrong dir or extension is given'
    for k, v in dict_fish.items():
        for fish_name in v:
            for f in sorted(dfs_files):
                if fish_name in f:
                    dict_files_per_fish[k].append(f)
    return dict_files_per_fish


def plot_3d_plot_tiomeofday_as_colr(fish_file, fish_nbr, flow_cat,
                                    fishshp, rivershp, out_save_dir):

    fig = plt.figure(figsize=(40, 20), dpi=75)
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

    in_df = pd.read_csv(fish_file, index_col=0, parse_dates=True)
    x_vals = in_df.Longitude.values
    y_vals = in_df.Latitude.values

    z_vals = in_df.index.to_pydatetime()
    dates_formatted = [pd.to_datetime(d) for d in z_vals]
    z_vals_ix = np.arange(0, len(z_vals), 1)

    bounds = {0: [0, 4], 1: [4, 10], 2: [10, 16], 3: [16, 22], 4: [22, 0]}
    clrs = ['blue', 'g',  'gold', 'red', 'blue']
    cmap = mcolors.ListedColormap(clrs)
    for ix, val in zip(in_df.index, in_df.index.hour):
        for k, v in bounds.items():
            if v[0] <= val <= v[1]:
                in_df.loc[ix, 'colors'] = clrs[k]
        if val < bounds[0][0]:
            in_df.loc[ix, 'colors'] = clrs[0]
        if val > bounds[4][0]:
            in_df.loc[ix, 'colors'] = clrs[4]

    ticks = [0, 4, 10, 16, 22, 24]

    ax.scatter3D(x_vals, y_vals, zs=z_vals_ix, zdir='z',
                 c=in_df.colors, alpha=0.65,
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
    norm = mcolors.BoundaryNorm(ticks, cmap.N)
    ax_legend = fig.add_axes([0.1725, 0.07525, 0.68, 0.0225], zorder=3)
    cb = mpl.colorbar.ColorbarBase(ax_legend, ticks=ticks,
                                   boundaries=ticks, norm=norm, cmap=cmap,
                                   orientation='horizontal')
    cb.set_label('Time_of_day_h', rotation=0)
    ax.set_aspect('auto')
    cb.ax.set_xticklabels([str(i) for i in ticks])

    ax.view_init(25, 275)

    ax.legend(loc=0)
    cb.draw_all()
    cb.set_alpha(1)
    plt.savefig(os.path.join(out_save_dir,
                             '3d_%s_%s_%s.png'
                             % (fish_nbr, flow_cat,
                                'Time_of_day_h')))
    plt.close()
    return

#%%


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


#%%


if __name__ == '__main__':

    START = timeit.default_timer()
    print('Program started at: ', time.asctime())

    dir_kmz_for_fish_names = (r'E:\Work_with_Matthias_Schneider'
                              r'\2018_11_26_tracks_fish_vemco\kmz')
    assert os.path.exists(dir_kmz_for_fish_names)

    fish_shp_path = (r'C:\Users\hachem\Desktop\Work_with_Matthias_Schneider'
                     r'\QGis_abbas\fish_pass.shp')
    assert os.path.exists(fish_shp_path)

    river_shp_path = (r'C:\Users\hachem\Desktop\Work_with_Matthias_Schneider'
                      r'\complere_river_shp\Altusried_Mesh_Boundary.shp')
    assert os.path.exists(river_shp_path)

    out_plots_dir = (r'C:\Users\hachem\Desktop'
                     r'\Work_with_Matthias_Schneider\out_plots_abbas')
    assert os.path.exists(out_plots_dir)
#    in_fish_files_dict = getFiles(r'C:\Users\hachem\Desktop\Work_with_Matthias_Schneider\out_plots_abbas\Filtered_data',
# '.csv')
    in_fish_files_dict = getFiles(
        r'C:\Users\hachem\Desktop\Work_with_Matthias_Schneider'
        r'\out_plots_abbas\df_fish_flow_combined_with_angles',
        '.csv', dir_kmz_for_fish_names)   #

    for fish_type in in_fish_files_dict.keys():
        for fish_file in in_fish_files_dict[fish_type]:
            print(fish_file)

            fish_nbr = fish_type + fish_file[-47:-42]
            # '_all_data_' + fish_file[-22:-17]  #    # fish_file[-32:-27]
            flow_cat = fish_file[-11:-5]
            # 'not_considered'  # fish_file[-11:-5]

            try:
                plot_3d_plot_tiomeofday_as_colr(fish_file, fish_nbr, flow_cat,
                                                fish_shp_path, river_shp_path,
                                                out_plots_dir)

                plot_3d_plot_flow_as_color(fish_file, fish_nbr, flow_cat,
                                           fish_shp_path, river_shp_path,
                                           out_plots_dir)

            except Exception as msg:
                print(msg)
                continue
#             break
#         break
    END = timeit.default_timer()

    print('Program ended at: ', time.asctime(),
          'Runtime was %0.4f seconds' % (END - START))
