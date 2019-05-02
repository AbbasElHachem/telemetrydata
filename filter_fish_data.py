# -*- coding: utf-8 -*-
"""
Created on 15-01-2019

@author: EL Hachem Abbas,
Institut fuer Wasser- und Umweltsystemmodellierung - IWS
"""
from __future__ import division

import math
import os
import time
import timeit

from PIL import Image
from matplotlib import style
from matplotlib.ticker import FormatStrFormatter
from scipy import spatial
from scipy import stats
from shapely.geometry import shape, Point
# from pandas_msgpack import to_msgpack
import psutil
import pyproj
import shapefile

import matplotlib as mpl

import matplotlib.colors as mcolors
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import pandas as pd


# mpl.use("Agg")
# import ffmpeg
# plt.rcParams['animation.ffmpeg_path'] = (r'C:\Users\hachem\Desktop\python3'
#                                         r'\WinPython-64bit-3.6.1.0Qt5\FFmpeg'
#                                         r'\bin\ffmpeg')
style.use('fivethirtyeight')

plt.ioff()
Image.MAX_IMAGE_PIXELS = 1000000000

print('\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime())
START = timeit.default_timer()  # to get the runtime of the program

dir_kmz_for_fish_names = (r'E:\Work_with_Matthias_Schneider'
                          r'\2018_11_26_tracks_fish_vemco\kmz')
assert os.path.exists(dir_kmz_for_fish_names)

orig_station_file = (r'E:\Work_with_Matthias_Schneider'
                     r'\2018_11_26_tracks_fish_vemco\stations.csv')
assert os.path.exists(orig_station_file)

main_data_dir = r'E:\Work_with_Matthias_Schneider\2018_11_26_tracks_fish_vemco'
assert os.path.exists(main_data_dir)
os.chdir(main_data_dir)

# out_plots_dir = r'E:\Work_with_Matthias_Schneider\out_plots_abbas'
out_plots_dir = (r'C:\Users\hachem\Desktop'
                 r'\Work_with_Matthias_Schneider\out_plots_abbas')
assert os.path.exists(out_plots_dir)

img_loc = r'E:\Work_with_Matthias_Schneider\GIS\orthoAll_small.jpg'
assert os.path.exists(img_loc)

shp_path = (r'E:\Work_with_Matthias_Schneider'
            r'\QGis_abbas\wanted_river_section.shp')
assert os.path.exists(shp_path)

fish_shp_path = (r'C:\Users\hachem\Desktop\Work_with_Matthias_Schneider'
                 r'\QGis_abbas\fish_pass.shp')
assert os.path.exists(fish_shp_path)

points_in_river = (r'C:\Users\hachem\Desktop'
                   r'\Work_with_Matthias_Schneider\df_in_river')
assert os.path.exists(points_in_river)

asci_grd_file = (r'C:\Users\hachem\Desktop\Work_with_Matthias_Schneider'
                 r'\2019_01_18_GridsFromHydraulicModelForIne'
                 r'\altusried_1m_copy.csv')
assert os.path.exists(asci_grd_file)

flow_data_file = (r'C:\Users\hachem\Desktop\Work_with_Matthias_Schneider'
                  r'\Flow data\q_summe_data_abbas.csv')
assert os.path.exists(flow_data_file)

fish_flow_files = (r'C:\Users\hachem\Desktop\Work_with_Matthias_Schneider'
                   r'\out_plots_abbas\Df_fish_flow')

simulated_flow_file = (r'C:\Users\hachem\Desktop\Work_with_Matthias_Schneider'
                       r'\2019_02_15_Altusried_hydraulics'
                       r'\Altusried_hydraulics_abbas.csv')
assert os.path.exists(simulated_flow_file)
# =============================================================================
#
# =============================================================================
# def epsg wgs84 and utm32
wgs82 = "+init=EPSG:4326"
utm32 = "+init=EPSG:32632"

# def font- and labelsize for plots
fontsize = 10
labelsize = 8

# def extent of the river image for plotting
extent = [10.2210163765499988, 10.2303021853499985,
          47.8146222938500003, 47.8224152275500032]

# for extracting periods for Barbel fish
periods_for_barbel = {'may_mid_june': ['2018-05-01 00:00:00.00',
                                       '2018-06-15 00:00:00.00'],
                      'mid_june_end_june': ['2018-06-15 00:00:00.00',
                                            '2018-06-30 00:00:00.00'],
                      'july_august': ['2018-07-01 00:00:00.00',
                                      '2018-08-31 00:00:00.00']}

# for extracting periods for Grayling fish
periods_for_grayling = {'mid_march_end_april': ['2018-03-15 00:00:00.00',
                                                '2018-04-30 00:00:00.00'],
                        'may_to_mid_may': ['2018-05-01 00:00:00.00',
                                           '2018-05-16 00:00:00.00'],
                        'mid_may_end_august': ['2018-05-16 00:00:00.00',
                                               '2018-08-31 00:00:00.00']}

# def HPE, RMSE, Velocity thresholds for filtering
hpe_thr = 1.35
rmse_thr = 0.35
vel_thr = 1.5


# =============================================================================
#
# =============================================================================

def list_all_full_path(ext, file_dir):
    import fnmatch
    """
    Purpose: To return full path of files in all dirs of a given folder with a
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


# =============================================================================
#
# =============================================================================


def get_file_names_per_fish_name(dir_fish_names_files):
    '''function to get all file names related to each fish '''
    fish_names = [name for name in os.listdir(dir_fish_names_files)
                  if os.path.isdir(os.path.join(dir_fish_names_files, name))]
    dict_fish = {k: [] for k in fish_names}
    for ix, key_ in enumerate(dict_fish.keys()):
        files_per_fish = os.listdir(os.path.join(
            dir_fish_names_files, fish_names[ix]))
        dict_fish[key_] = files_per_fish
    return dict_fish
# =============================================================================
#
# =============================================================================


def getFiles(data_dir_, file_ext_str):
    ''' function to get files based on dir and fish name'''
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
# =============================================================================
#
# =============================================================================


def read_OrigStn_DF(df_stn_orig_file):
    ''' function to read fixed station data'''
    df_orig_stn = pd.read_csv(df_stn_orig_file, sep=',',
                              usecols=['StationName', 'Longitude', 'Latitude'])
    return df_orig_stn
# =============================================================================
#
# =============================================================================


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
# =============================================================================
#
# =============================================================================


def convert_coords_fr_wgs84_to_utm32_(epgs_initial_str, epsg_final_str,
                                      first_coord, second_coord):
    '''fct to convert points from wgs 84 to utm32'''
    initial_epsg = pyproj.Proj(epgs_initial_str)
    final_epsg = pyproj.Proj(epsg_final_str)
    x, y = pyproj.transform(initial_epsg, final_epsg,
                            first_coord, second_coord)
    return x, y
# =============================================================================
#
# =============================================================================


def check_if_points_in_polygone(shp_path, df_points):
    ''' function to check if points are in the river or not'''
    assert shp_path
    shp_river = shapefile.Reader(shp_path)
    shapes = shp_river.shapes()
    polygon = shape(shapes[0])

    def check(lon_coord, lat_coord):
        ''' check is point in river or not'''
        point = Point(lon_coord, lat_coord)
        return polygon.contains(point)

    for ix, x, y in zip(df_points.index, df_points['Longitude'].values,
                        df_points['Latitude'].values):
        df_points.loc[ix, 'In_River'] = check(x, y)
    return df_points
# =============================================================================
#
# =============================================================================


def find_fish_in_river(df, river_shp, out_save_dir, fish_nbr, fish_type=None):
    '''a function to find all points in river and
        save results tp new data frame '''
    if fish_type is not None:
        out_save_dir = os.path.join(out_plots_dir, fish_type)
        if not os.path.exists(out_save_dir):
            os.mkdir(out_save_dir)

    df_save_name = os.path.join(out_save_dir,
                                r'%s_points_in_river.csv' % fish_nbr)

    if not os.path.exists(df_save_name):
        df_points_in_river = check_if_points_in_polygone(river_shp, df)
        df_points_only_in_river = df_points_in_river[df_points_in_river[
            'In_River'] == True]
        df_points_only_in_river.to_csv(df_save_name, sep=';')
        return df_points_only_in_river
    else:
        df_points_only_in_river = pd.read_csv(df_save_name,
                                              sep=';', index_col=0)
        return df_points_only_in_river


# =============================================================================
#
# =============================================================================


def multi_proc_shp_inter(in_fish_files_dict):
    ''' a multiprocessed function to intersect points and river shpfile'''

    def save_all_df_in_river(fish_file):
        ''' intersect points and shapefile river'''
        print(fish_file)
        fish_tag_nbr = fish_file[-9:-4]
        if not os.path.exists(os.path.join(points_in_river,
                                           r'%s_points_in_river.csv'
                                           % fish_tag_nbr)):
            print('working for file, ', fish_tag_nbr)
            df_orig = readDf(fish_file)
            delta_x_y = calculate_fish_velocity(df_orig)
            delta_x_y_below_2ms = use_Variable_below_thr_keep_first_point(
                delta_x_y, 'Velocity', 1.5)
            find_fish_in_river(delta_x_y_below_2ms, shp_path, fish_tag_nbr)
            print('done saving data for: ', fish_tag_nbr)
        else:
            print('skipping file, ', fish_file)
        return

    cpu_count = psutil.cpu_count() - 3
    mp_pool = mp.Pool(cpu_count)
    mp_pool.map(save_all_df_in_river,
                (fish_file for fish_files in
                 list(in_fish_files_dict.values())
                 for fish_file in fish_files))
    return


# =============================================================================
#
# =============================================================================


def readDf_points_in_river(df_file):
    ''' read on df in river and adjust index and selct columns'''
    df = pd.read_csv(df_file, sep=';', index_col=0, infer_datetime_format=True)
    time_fmt = '%Y-%m-%d %H:%M:%S.%f'
    try:
        df.index = pd.to_datetime(df.index, format=time_fmt)
    except ValueError:
        df.index = [ix.replace('.:', '.') for ix in df.index]
        df.index = pd.to_datetime(df.index, format=time_fmt)
    df.drop(['delta_x', 'delta_y', 'time_delta', 'In_River'],
            axis=1, inplace=True)
    return df

# =============================================================================
# =============================================================================
# # START PART 2
# =============================================================================
# =============================================================================


def calculate_distance_2_points(deltax, deltay):
    ''' fct to calculate distance between two arrays of coordinates'''
    return np.sqrt((deltax ** 2) + (deltay ** 2))
# =============================================================================
#
# =============================================================================


def calculate_fish_velocity(df_fish):
    ''' function to calculate travel velocity between subsequent locations'''

    x, y = convert_coords_fr_wgs84_to_utm32_(wgs82, utm32,
                                             df_fish['Longitude'].values,
                                             df_fish['Latitude'].values)

    df_utm32 = pd.DataFrame(index=df_fish.index,
                            data={'delta_x': x, 'delta_y': y})

    delta_x_y = df_utm32.diff()
    delta_x_y.dropna(inplace=True)
    delta_x_y['Time'] = delta_x_y.index
    delta_x_y['time_delta'] = delta_x_y.Time.diff() / pd.Timedelta('1s')

    (delta_x_y['Longitude'], delta_x_y['Latitude']) = (
        convert_coords_fr_wgs84_to_utm32_(utm32, wgs82, x[1:], y[1:]))

    delta_x_y['distance'] = calculate_distance_2_points(
        delta_x_y['delta_x'], delta_x_y['delta_y'])

    delta_x_y['Velocity'] = (delta_x_y['distance'].values /
                             delta_x_y['time_delta'].values)

    delta_x_y['Point_x'], delta_x_y['Point_y'] = x[1:], y[1:]
    delta_x_y['HPE'], delta_x_y['RMSE'] = (df_fish['HPE'].values[1:],
                                           df_fish['RMSE'].values[1:])
    return delta_x_y
# =============================================================================
#
# =============================================================================


def use_Variable_below_thr_keep_first_point(df_fish, var_name, var_thr):
    ''' use a filter based on Variable threshold keep first point'''
    df_fish = df_fish.copy()
    df_fish = df_fish[df_fish[var_name] <= var_thr]
    return df_fish
# =============================================================================
#
# =============================================================================


def use_Variable_below_thr_two_var(df_fish, var1_name, var1_thr,
                                   var2_name, var2_thr):
    ''' use a filter based on two different Variables thresholds'''
    df_fish = df_fish.copy()
    df_fish = df_fish[(df_fish[var1_name] <= var1_thr) &
                      (df_fish[var2_name] <= var2_thr)]
    return df_fish
# =============================================================================
#
# =============================================================================


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
# =============================================================================
#
# =============================================================================


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
# =============================================================================
#
# =============================================================================


def use_Variable_abv_thr(df_fish, var_name, var_thr):
    ''' use a filter based on Variable threshold'''
    df_fish = df_fish.copy()
    df_fish = df_fish[df_fish[var_name] >= var_thr]
    return df_fish


# =============================================================================
#
# =============================================================================


def generate_kml_files(df_in_river, time_steps_lim):
    ''' fct to generate Kml/Kmz google earth files'''
    import simplekml
    kml = simplekml.Kml()
    i = 0
    for ix, lon, lat in zip(df_in_river.index.values,
                            df_in_river.Longitude.values,
                            df_in_river.Latitude.values):

        while i < time_steps_lim:
            print(i)
            kml.newpoint(name=str(ix), coords=[(lon, lat)])
            i += 1
    kml.save(os.path.join(out_plots_dir, 'fish_mvt.kmz'))

# =============================================================================
# =============================================================================
# # END PART 2, START PART 3
# =============================================================================
# =============================================================================


def savefig(fig_name, out_dir):
    ''' fct to save fig based on name and outdir'''
    return plt.savefig(os.path.join(out_dir, '_%s.png' % fig_name),
                       frameon=True, papertype='a4',
                       bbox_inches='tight', pad_inches=.2)

# =============================================================================
#
# =============================================================================


def transform_variables_to_uniform_marginals(df_fish, VarA, VarB):
    '''
        transform variables to uniform marginals for Copulas
        this is used to plot the pure dependence between two
        variables
    '''
    varx = df_fish[VarA].values
    vary = df_fish[VarB].values
    xvals, yvals = np.squeeze(varx), np.squeeze(vary)

    ranks_R1i = (len(xvals) - stats.rankdata(xvals) + 0.5) / len(xvals)
    ranks_S1i = (len(yvals) - stats.rankdata(yvals) + 0.5) / len(yvals)

    plt.scatter(ranks_R1i, ranks_S1i, alpha=0.5)
    _, (ax1, ax2) = plt.subplots(
        1, 2, sharey=True, sharex=True)  # , projection='3d')
    ax1.scatter(ranks_S1i, ranks_R1i, c='b', marker='.', alpha=0.5, s=0.5)

    ax2.hist2d(ranks_S1i, ranks_R1i, bins=10, normed=True)
    ax1.set_xlabel('Normed Ranked %s' % VarA)
    ax2.set_xlabel('Normed Ranked %s' % VarA)
    ax1.set_ylabel('Normed Ranked %s' % VarB)
    savefig('marignals_var_%s_var_%s' % (VarA, VarB), out_plots_dir)
    return


# =============================================================================
#
# =============================================================================


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


# make customized colormap
c = mcolors.ColorConverter().to_rgb
rvb = make_colormap([c('blue'), c('lightblue'), c('c'), 0.33,
                     c('green'), c('yellow'), c('gold'), 0.66,
                     c('orange'), c('red')])
# =============================================================================
#
# =============================================================================


def plot_histogram(df_fish, fish_nbr, variable_to_plt,
                   var_thr=None, use_log=False):
    ''' fct to plot histogram for a variable'''
    _, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=100)
    vals = df_fish[variable_to_plt].values
    if var_thr is not None:
        vals = vals[vals <= var_thr]
    if use_log:
        vals = np.log(vals)
    bins = np.arange(vals.min(), vals.max(), 0.5)
    center = (bins[:-1] + bins[1:]) / 2
    hist, bins2 = np.histogram(vals, bins=bins, density=False)
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
    savefig('%s_distribution_%s' % (variable_to_plt, fish_nbr), out_plots_dir)
    plt.clf()
    plt.close()
    return


# =============================================================================
#
# =============================================================================

def plot_img(img_loc, ax, img_transparancy=0.25):
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

# =============================================================================
#
# =============================================================================


def plot_Var_values(img_loc, orig_station_file, fish_file,
                    fish_type_nbr, var_to_plt, var_bounds):
    ''' fct to plot any variable (Velocity, HPe, RMSE) based on input file '''
    in_orig_stn_df = read_OrigStn_DF(orig_station_file)
    try:
        df_orig = readDf(fish_file)
        df_fish = calculate_fish_velocity(df_orig)
    except Exception:
        df_fish = readDf_points_in_river(fish_file)
    if var_to_plt == 'Velocity':
        df_fish = use_Variable_below_thr_keep_first_point(
            df_fish, var_to_plt, 1.5)

    norm = mcolors.BoundaryNorm(boundaries=var_bounds, ncolors=256)
    fig, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=800)

    plot_img(img_loc, ax)
    pcm = ax.scatter(df_fish['Longitude'].values,
                     df_fish['Latitude'].values,
                     c=df_fish[var_to_plt].values, s=0.05, alpha=0.5,
                     marker=',', cmap='RdBu_r', norm=norm,
                     vmin=df_fish[var_to_plt].values.min(),
                     vmax=df_fish[var_to_plt].values.max())
    ax.scatter(in_orig_stn_df['Longitude'].values,
               in_orig_stn_df['Latitude'].values,
               s=0.08, marker='x', c='k', label='Ref Stations', alpha=0.75)

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
    savefig('%s_below_2ms_vls2_fish_%s' %
            (var_to_plt, fish_type_nbr), out_plots_dir)
    plt.clf()
    plt.close()
    return
# =============================================================================
#
# =============================================================================


def plot_variables(data_loc, Velocity=False, RMSE=False, HPE=False):
    ''' fct used to plot scatter plot of all data files in given directory'''
    in_fish_files_dict = getFiles(data_loc, '.csv')
    velocity_bounds = np.array([0, 0.25, 0.5, 1, 1.5, 2., 2.5, 3])
    HPE_bounds = np.array([0, 3, 5, 10, 15, 20, 21])
    RMSE_bounds = np.array([0, .01, .03, .05, .07, .1, 0.25, 1.0])
    for fish_type in in_fish_files_dict.keys():
        print('fish type is:', fish_type)
        for fish_file in in_fish_files_dict[fish_type]:
            try:
                fish_tag_nbr = fish_file[-9:-4]
                assert '4' in fish_tag_nbr
            except Exception:
                fish_tag_nbr = fish_file[-25:-20]
                assert '4' in fish_tag_nbr
            print('fish file is: ', fish_file)
            if Velocity:
                plot_Var_values(img_loc, orig_station_file, fish_file,
                                (fish_tag_nbr + '_' + fish_type),
                                'Velocity', velocity_bounds)
            if RMSE:
                plot_Var_values(img_loc, orig_station_file, fish_file,
                                (fish_tag_nbr + '_' + fish_type),
                                'RMSE', RMSE_bounds)
            if HPE:
                plot_Var_values(img_loc, orig_station_file, fish_file,
                                (fish_tag_nbr + '_' + fish_type),
                                'HPE', HPE_bounds)
            print('done saving data for: ', fish_file)
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


def plot_heatmapt_fish_loc(df_fish, fish_nbr, fish_type='',
                           plt_img=False, weights=None):
    ''' fct to plot heatmap of a fish '''
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.ticker import FormatStrFormatter

    plt.rcParams['agg.path.chunksize'] = 10000

    x = df_fish['Longitude'].values
    y = df_fish['Latitude'].values
    fig, (ax1, ax2) = plt.subplots(
        1, 2, sharey=True, sharex=True,
        figsize=(18, 12), dpi=400)
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
    ax1.set_title('Filtered data for fish _%s_%s '
                  % (fish_nbr, fish_type), fontsize=fontsize)
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

    img[img == 0] = np.nan

    im = ax2.imshow(np.log(img), extent=extent_ht, origin='lower', cmap=rvb)
    ax2.set_title("Bi-dimensional Log Histogram Lon-Latt fish %s"
                  % (fish_nbr),  # fish_type),
                  fontsize=fontsize)

    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.08)

    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=7)
    if plt_img:
        plot_img(img_loc, ax1), plot_img(img_loc, ax2)
    savefig('heatmap_fish_%s_without_back_%s' % (fish_nbr, fish_type),
            out_plots_dir)
    plt.close()

    return
# =============================================================================
#
# =============================================================================


def plot_loc_time_vls(df_fish, fish_type_nbr):
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


def select_df_within_period(df, start, end):
    ''' a function to select df between two dates'''
    mask = (df.index > start) & (df.index <= end)
    df_period = df.loc[mask]
    return df_period


# =============================================================================
#
# =============================================================================
def filter_plot_fish_data_by_period(df_fish, fish_type_nbr, fish_type,
                                    start_period_one, end_period_one,
                                    start_period_two, end_period_two,
                                    start_period_three, end_period_three,
                                    plot_heat_maps_per_period=False,
                                    use_time_weights=False):
    ''' function to seperate and plot data based on 3 defined periods'''
    if use_time_weights:
        df_fish = calculate_weights_for_heatmaps(df_fish)
    df_period_may_mid_june = select_df_within_period(df_fish,
                                                     start_period_one,
                                                     end_period_one)

    df_period_mid_june_end_june = select_df_within_period(df_fish,
                                                          start_period_two,
                                                          end_period_two)

    df_period_july_august = select_df_within_period(df_fish,
                                                    start_period_three,
                                                    end_period_three)
    if plot_heat_maps_per_period:

        if use_time_weights:
            weights_may_mid_june = df_period_may_mid_june.Weights
            weights_mid_june_end_june = df_period_mid_june_end_june.Weights
            weights_july_august = df_period_july_august.Weights
        else:
            (weights_may_mid_june, weights_mid_june_end_june,
             weights_july_august) = None, None, None

        plot_heatmapt_fish_loc(df_period_may_mid_june,
                               fish_type_nbr,
                               '_period_may_mid_june',
                               plt_img=True,
                               weights=weights_may_mid_june)

        plot_heatmapt_fish_loc(df_period_mid_june_end_june,
                               fish_type_nbr,
                               '_period_mid_june_end_june',
                               plt_img=True,
                               weights=weights_mid_june_end_june)

        plot_heatmapt_fish_loc(df_period_july_august,
                               fish_type_nbr,
                               '_period_july_august',
                               plt_img=True,
                               weights=weights_july_august)
        return
    else:
        _, ax = plt.subplots(1, 1, figsize=(10, 8), dpi=400)

        ax.scatter(df_period_may_mid_june['Longitude'].values,
                   df_period_may_mid_june['Latitude'].values,
                   c='darkblue', s=0.05, alpha=0.05,
                   marker='+',
                   label='darkblue_period_may_mid_june ')
        ax.scatter(df_period_mid_june_end_june['Longitude'].values,
                   df_period_mid_june_end_june['Latitude'].values,
                   c='darkred', s=0.05, alpha=0.05,
                   marker='o',
                   label='darkred_period_mid_june_end_june ')
        ax.scatter(df_period_july_august['Longitude'].values,
                   df_period_july_august['Latitude'].values,
                   c='darkgreen', s=0.05, alpha=0.05,
                   marker='*',
                   label='darkgreen_period_july_august ')
        plot_img(img_loc, ax)
        ax.set_xlabel('Longitude', fontsize=fontsize)
        ax.set_ylabel('Latitude', fontsize=fontsize)
        ax.tick_params(axis='x', labelsize=labelsize)
        ax.tick_params(axis='y', labelsize=labelsize)
        ax.set_title('Locations for Fish: %s %s'
                     % (fish_type_nbr, fish_type),
                     fontsize=fontsize, y=0.99)
        plt.legend(loc=0, fontsize=fontsize)
        plt.grid(alpha=0.15)
        savefig('%s_fish_%s_%s'
                % ('Locations_for_', fish_type_nbr, fish_type),
                out_plots_dir)
        plt.clf()
        plt.close()

        return


# =============================================================================
#
# =============================================================================
def save_fish_per_period(df_fish, fish_nbr,
                         periods_dict_names_dates):
    '''
        given a fish dataframe for a fish nbr with a fish type
        given a list containing the periods in tuples (start, end)
        given a list of period names
        extract for every fish the data for different periods seperatly
    '''
    for per_name in periods_dict_names_dates.keys():
        per_dtes = periods_dict_names_dates[per_name]

        df_period = select_df_within_period(df_fish, per_dtes[0], per_dtes[1])
        df_period = df_period.loc[:, ['Longitude', 'Latitude']]
        if df_period.values.shape[0] > 0:
            print(df_period)
            df_period.to_csv(os.path.join(out_plots_dir,
                                          r'filtered_data_%s_HPE_RMSE_Vel_%s.csv'
                                          % (fish_nbr, per_name)))
    return
# =============================================================================
# =============================================================================
# # END PART 3, START PART 4
# =============================================================================
# =============================================================================


def calculate_angle_between_two_positions(df_fish, xname='Longitude',
                                          yname='Latitude'):
    ''' calculate angle between two successive positions '''

    x_vals, y_vals = df_fish[xname].values, df_fish[yname].values

    angles_degs = [np.math.degrees(np.math.atan2(y_vals[i] - y_vals[i - 1],
                                                 x_vals[i] - x_vals[i - 1]))
                   for i in range(1, df_fish.values.shape[0])]
    angles_degs.insert(0, np.nan)
    df_fish['fish_angle'] = angles_degs  # _%s_%s' % (xname, yname)
    return df_fish

# =============================================================================
#
# =============================================================================


def calculate_angle_between_flow_vectors(df_fish_flow,
                                         flow_cat_str):
    ''' calculate angle between flow vectors Vx, Vy '''

    xname, yname = 'velX_%s' % flow_cat_str, 'velY_%s' % flow_cat_str
    x_vals, y_vals = df_fish_flow[xname].values, df_fish_flow[yname].values

    angles_degs = [np.math.degrees(np.math.atan2(y_vals[i] - 0,
                                                 x_vals[i] - 0))
                   for i in range(1, df_fish_flow.values.shape[0])]
    angles_degs.insert(0, np.nan)
    df_fish_flow['flow_angle'] = angles_degs  # _%s_%s' % (xname, yname)
    return df_fish_flow


# =============================================================================
#
# =============================================================================


def plot_angles_two_positions(lons, lats, angles, fish_type_nbr):
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
# =============================================================================
#
# =============================================================================


def aggregate_values_per_grid_cell(df_fish, vel_col_name='Velocity',
                                   angle_col_name='fish_angle'):
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
    if 'fish_angle' not in df_fish.columns:
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


# =============================================================================
#
# =============================================================================
def plot_agg_grid_vls(grdx, grdy, var_vls, fish_nbr, var_name):
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

def save_cat_flow_data(obsv_flow_file,
                       simulated_flow_file,
                       df_fish,
                       fish_type_nbr):
    '''
        a function to read the observed flow data and
        observed fish locations, find for every position,
        based on time of measure the corresponding observed
        flow and categorize the output based on different flow
        categories. Save the resulted dataframes
    Input:
        obsv_flow_file: observed river flow values in m3/s (
        simulated_flow_file: simualted hydraulics, different scenarios
        df_fish: filtered observed fish dataframe
        fish_type_nbr: type and ID of fish, for saving output
    Output:
        For each Flow categorie (20, 30, 40, 50, 60, 80) add
        or every observed fish position, the corresponding Hydraulics
        variables, the output is for each flow categorie, the corresponding
        Fish variables in a seperate df
    '''
    # read observed flow file
    df_flow = pd.read_csv(obsv_flow_file, index_col=0, parse_dates=True)
    # find start and end time of fish observations
    start_fish_t, end_fish_t = df_fish.index[0], df_fish.index[-1]
    # reduce the flow values to only observed fish values
    df_flow = select_df_within_period(df_flow, start_fish_t, end_fish_t)

    # read simulated flow, depth data
    sim_flow = pd.read_csv(simulated_flow_file, sep=',')
    # TODO MAKE ME FASTER
    for _, ix_f in enumerate(df_fish.index):
        # get nearest index in flow df to fish index
        idx = df_flow.index[df_flow.index.get_loc(ix_f, method='nearest')]
        flow_vl = df_flow.loc[idx, :].values
        print(ix_f, idx, flow_vl)
        # categorize values based on hydraulics
        if flow_vl <= 15:
            df_fish.loc[ix_f, 'Flow_Cat'] = 10  # cat 10
        if 15 < flow_vl <= 25:
            df_fish.loc[ix_f, 'Flow_Cat'] = 20  # cat 20
        if 25 < flow_vl <= 35:
            df_fish.loc[ix_f, 'Flow_Cat'] = 30
        if 35 < flow_vl <= 45:
            df_fish.loc[ix_f, 'Flow_Cat'] = 40
        if 45 < flow_vl <= 55:
            df_fish.loc[ix_f, 'Flow_Cat'] = 50
        if 55 < flow_vl <= 65:
            df_fish.loc[ix_f, 'Flow_Cat'] = 60
        if 65 < flow_vl:
            df_fish.loc[ix_f, 'Flow_Cat'] = 80
    # keep only needed columns
    df_fish_flow = df_fish.loc[:, ['Longitude', 'Latitude', 'Velocity',
                                   'HPE', 'RMSE', 'Flow_Cat']]
    # transform coordinates
    (df_fish_flow['x_fish'],
     df_fish_flow['y_fish']) = convert_coords_fr_wgs84_to_utm32_(
        wgs82, utm32, df_fish_flow.Longitude.values,
        df_fish_flow.Latitude.values)

    for ix, xf, yf in zip(df_fish_flow.index,
                          df_fish_flow['x_fish'].values,
                          df_fish_flow['y_fish'].values):
        flow_val = str(int(df_fish_flow.loc[ix, 'Flow_Cat']))
        for col in sim_flow.columns:
            if flow_val in col:
                for igx, grdx, grdy in zip(sim_flow.index, sim_flow.x.values,
                                           sim_flow.y.values):
                    # idea: find closest grid point to fish location
                    if (math.isclose(xf, grdx, rel_tol=10e-7) and
                            math.isclose(yf, grdy, rel_tol=10e-8) is True):

                        print(igx, xf, grdx, yf, grdy, col)
                        df_fish_flow.loc[ix, 'index_of_grid_node'] = igx
                        df_fish_flow.loc[ix, 'X_of_grid_node'] = \
                            sim_flow.loc[igx, 'x']
                        df_fish_flow.loc[ix, 'Y_of_grid_node'] = \
                            sim_flow.loc[igx, 'y']
                        df_fish_flow.loc[ix, 'Z_of_grid_node'] = \
                            sim_flow.loc[igx, 'z']
                        df_fish_flow.loc[ix, 'depth_%s' % flow_val] = \
                            sim_flow.loc[igx, 'depth_%s' % flow_val]

                        df_fish_flow.loc[ix, 'velX_%s' % flow_val] = \
                            sim_flow.loc[igx, 'velX_%s' % flow_val]
                        df_fish_flow.loc[ix, 'velY_%s' % flow_val] = \
                            sim_flow.loc[igx, 'velY_%s' % flow_val]
                        # if magnitud not in columns, calcualte flow mag
                        if 'velM_%s' % flow_val in sim_flow.columns:
                            df_fish_flow.loc[ix, 'velM_%s' % flow_val] = \
                                sim_flow.loc[igx, 'velM_%s' % flow_val]

                        break
    # extract every dataframe categorie to save output
    flow_cat_10 = df_fish_flow[df_fish_flow['Flow_Cat'] == 10]
    flow_cat_20 = df_fish_flow[df_fish_flow['Flow_Cat'] == 20]
    flow_cat_30 = df_fish_flow[df_fish_flow['Flow_Cat'] == 30]
    flow_cat_40 = df_fish_flow[df_fish_flow['Flow_Cat'] == 40]
    flow_cat_50 = df_fish_flow[df_fish_flow['Flow_Cat'] == 50]
    flow_cat_60 = df_fish_flow[df_fish_flow['Flow_Cat'] == 60]
    flow_cat_80 = df_fish_flow[df_fish_flow['Flow_Cat'] == 80]

    flow_cat_list = [flow_cat_10, flow_cat_20, flow_cat_30, flow_cat_40,
                     flow_cat_50, flow_cat_60, flow_cat_80]
    for flow_cat in flow_cat_list:
        if flow_cat.values.shape[0] > 0:
            flow_cat.to_csv(os.path.join(out_plots_dir,
                                         'fish_%s_with_flow_data_cat_%s_.csv')
                            % (fish_type_nbr,
                               str(int(flow_cat.Flow_Cat.values[0]))))

    return df_fish_flow


# =============================================================================
#
# =============================================================================

def find_diff_fish_and_flow_direction(fish_file, fish_type_nbr, flow_cat):
    '''
        a function used to plot the difference between flow and fish
        angles
    '''
    in_df = pd.read_csv(fish_file, index_col=0)

    flow_val = flow_cat[-2:]
    in_df.dropna(axis=1, inplace=True)
    in_df = calculate_angle_between_two_positions(in_df, 'x_fish', 'y_fish')
    in_df = calculate_angle_between_flow_vectors(in_df, flow_val)
    in_df['angle_diff'] = np.mod(in_df['fish_angle'] -
                                 in_df['flow_angle'] + 180, 360) - 180
    in_df.to_csv(os.path.join(out_plots_dir,
                              'fish_%s_with_flow_data_and_angles_cat_%s_.csv')
                 % (fish_type_nbr, str(flow_cat)))
    return in_df
# =============================================================================
#
# =============================================================================


def plot_difference_in_angle(in_df_fish_flow, fish_type_nbr, flow_cat,
                             angle_col_name_str):
    if not isinstance(in_df_fish_flow, pd.DataFrame):
        in_df_fish_flow = pd.read_csv(in_df_fish_flow, index_col=0)

    ticks = [-180, -135, -90, -45, 0, 45, 90, 135, 180]

    clrs = ['red', 'gold', 'gold', 'green', 'green', 'gold', 'gold', 'red']
    cmap = mcolors.ListedColormap(clrs)

    bounds = {0: [-180, -135], 1: [-135, -90], 2: [-90, -45], 3: [-45, 0],
              4: [0, 45], 5: [45, 90], 6: [90, 135], 7: [135, 180]}

    for ix, val in zip(in_df_fish_flow.index,
                       in_df_fish_flow[angle_col_name_str].values):
        for k, v in bounds.items():
            if v[0] <= val <= v[1]:
                in_df_fish_flow.loc[ix, 'colors'] = clrs[k]
        if val < bounds[0][0]:
            in_df_fish_flow.loc[ix, 'colors'] = clrs[0]
        if val > bounds[7][1]:
            in_df_fish_flow.loc[ix, 'colors'] = clrs[7]

    if in_df_fish_flow.colors[1:].isna().sum() >= 1:
        print('nan values', in_df_fish_flow.colors[1:].isna().sum())
        in_df_fish_flow.colors[1:].fillna(value='k', inplace=True)
    vmin, vmax, extend = -180, 180, 'neither'

    fig, ax0 = plt.subplots(1, 1, figsize=(20, 10), dpi=100)

    ax0.scatter(in_df_fish_flow.Longitude[1:], in_df_fish_flow.Latitude[1:],
                c=in_df_fish_flow.colors[1:],
                cmap=cmap, s=0.35, alpha=0.75, marker=',',
                vmin=vmin, vmax=vmax)

    norm = mcolors.BoundaryNorm(ticks, cmap.N)
    ax_legend = fig.add_axes([0.1725, 0.0025, 0.68, 0.0225], zorder=3)
    cbar = mpl.colorbar.ColorbarBase(ax_legend, extend=extend, ticks=ticks,
                                     boundaries=ticks, norm=norm, cmap=cmap,
                                     orientation='horizontal')
    # 'Angle Difference Fish-Flow Deg'
    cbar.set_label(angle_col_name_str + ' in deg',
                   fontsize=fontsize)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_alpha(1), cbar.draw_all()
    ax0.set_title(angle_col_name_str + ' Deg %s for Flow Cat %s'
                  % (fish_type_nbr, flow_cat),
                  fontsize=fontsize)
    ax0.set_xlim([10.222, 10.228]), ax0.set_ylim([47.8175, 47.8205])
    ax0.set_xticks([10.222, 10.223, 10.224, 10.225, 10.226, 10.227, 10.228])
    ax0.set_yticks([47.8175, 47.8180, 47.8185, 47.8190,
                    47.8195, 47.8200, 47.8205])
    ax0.set_xlabel('Longitude', fontsize=fontsize)
    ax0.set_ylabel('Latitude', fontsize=fontsize)
    ax0.tick_params(axis='x', labelsize=labelsize)
    ax0.tick_params(axis='y', labelsize=labelsize)

    ax0.xaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    ax0.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    ax0.grid(alpha=0.15)

    plt.savefig(os.path.join(out_plots_dir,
                             '%s_%s_flow_cat_%s.png'
                             % (angle_col_name_str, fish_type_nbr, flow_cat)),
                frameon=True, papertype='a4',
                bbox_inches='tight', pad_inches=.2)
    plt.close()
    plt.clf()
    del in_df_fish_flow
    return

# =============================================================================
#
# =============================================================================


from scipy.spatial import distance


def find_distances(pointPx, pointPy, df):
    # find distance from point P to all surroudning points
    df['distances'] = np.sqrt(np.square((df['X_of_grid_node'].values - pointPx) +
                                        (df['Y_of_grid_node'].values - pointPy)))
    return df


def closest_node(node, nodes):
    return distance.cdist([node], nodes, 'euclidean')


def get_surrounding_nodes(x_center, y_center, xall, yall):
    right_point = (x_center + 1, y_center)
    upper_point = (x_center, y_center + 1)
    left_point = (x_center - 1, y_center)
    bottom_point = (x_center, y_center - 1)

    surrounding_nodes_coords = [right_point, upper_point,
                                left_point, bottom_point]
    xoords_wanted = [surrounding_nodes_coords[i][0]
                     for i in range(len(surrounding_nodes_coords))
                     if surrounding_nodes_coords[i][0] in xall]
    yoords_wanted = [surrounding_nodes_coords[i][1]
                     for i in range(len(surrounding_nodes_coords))
                     if surrounding_nodes_coords[i][1] in yall]
    return (xoords_wanted, yoords_wanted, surrounding_nodes_coords)
#==============================================================================
#
#==============================================================================


from sklearn.neighbors import NearestNeighbors


def calc_max_gradient_direct(fish_flow_file, flow_cat, fish_nbr):
    '''
    a function used to calculate between every position
    and closest 4 positions the difference in the
    depth gradient and flow velocity
    find the maximum difference and add it to the df

    https://stackoverflow.com/questions/53028514/
    calculate-distance-from-one-point-to-all-others
    '''
    fish_flow_df = pd.read_csv(fish_flow_file, sep=',', index_col=0,
                               parse_dates=True, engine='c')
    flow_val = flow_cat[-2:]
    depth_var = 'depth_%s' % flow_val
    flow_var = 'velM_%s' % flow_val

#     fish_flow_df_all = pd.read_feather(fish_flow_file, use_threads=4)
#     fish_flow_df_all.set_index('Time', inplace=True)

    # split df into 10 parts
#     dfs_list = np.array_split(fish_flow_df_all, 10)
#     dfs_all_end = []

#     for fish_flow_df in dfs_list:
    x_grid = fish_flow_df.X_of_grid_node.values.ravel()
    y_grid = fish_flow_df.Y_of_grid_node.values.ravel()

    print('calculting for', fish_flow_file)

#             points_tree1 = spatial.cKDTree(
#                 np.c_[nodes_coords1], compact_nodes=False)

#     print('done constructing tree')
    for ix, x0, y0 in zip(fish_flow_df.index, x_grid, y_grid):
        print(ix, x0, y0)

        depth_point0 = fish_flow_df.loc[ix, depth_var]
        flow_vel_point0 = fish_flow_df.loc[ix, flow_var]
#         plt.scatter(x_grid, y_grid, c='b')
#         plt.scatter(x0, y0, c='r')
#         plt.show()
        df_with_distances = find_distances(x0, y0, fish_flow_df)
        try:
            df_to_use = df_with_distances[(0 < df_with_distances.distances) &
                                          (df_with_distances.distances <= .999)]
            xnear = df_to_use.X_of_grid_node.values.ravel()
            ynear = df_to_use.Y_of_grid_node.values.ravel()
            nodes_coords1 = np.array([(x, y) for x, y in zip(xnear, ynear)])
            neigh = NearestNeighbors(n_neighbors=4)
    #         neigh = NearestNeighbors(radius=0.998)
            nbrs = neigh.fit(nodes_coords1)
            distances, indices = nbrs.kneighbors([[x0, y0]])
            indices = indices[0]
#         distances, indices = nbrs.radius_neighbors([[x0, y0]])

#             xss = []
#             yss = []
#             for idc in indices[0]:
#                 xs = fish_flow_df.iloc[idc, :]['X_of_grid_node']
#                 ys = fish_flow_df.iloc[idc, :]['Y_of_grid_node']
#                 xss.append(xs)
#                 yss.append(ys)
#             plt.scatter(x0, y0, c='r')
#             plt.scatter(xss, yss, c='b', marker='.')
#             plt.show()

            diff_in_grds_lst, diff_in_vel_lst = [], []
            for idc in indices:
                point_i_depth = fish_flow_df.iloc[idc, :][depth_var]
                point_i_flow_mag = fish_flow_df.iloc[idc, :][flow_var]

                diff_in_grds_lst.append(
                    np.abs(depth_point0 - point_i_depth))
                diff_in_vel_lst.append(
                    np.abs(flow_vel_point0 - point_i_flow_mag))

            (_, depth_point_id) = (np.max(diff_in_grds_lst),
                                   indices[np.argmax(diff_in_grds_lst)])
            (_, vel_point_id) = (np.max(diff_in_vel_lst),
                                 indices[np.argmax(diff_in_vel_lst)])

            (x_d, y_d) = (fish_flow_df.iloc[depth_point_id, :].x_fish,
                          fish_flow_df.iloc[depth_point_id, :].y_fish)
            (x_v, y_v) = (fish_flow_df.iloc[vel_point_id, :].x_fish,
                          fish_flow_df.iloc[vel_point_id, :].y_fish)

            angle_fish_max_depth_grd = np.math.degrees(
                np.math.atan2((y_d - y0), (x_d - x0)))
            angle_fish_max_vel_grd = np.math.degrees(
                np.math.atan2((y_v - y0), (x_v - x0)))

            fish_flow_df.loc[ix,
                             'Angle_swim_direction_and_max_%s_gradient_difference'
                             % depth_var] = angle_fish_max_depth_grd
            fish_flow_df.loc[ix,
                             'Angle_swim_direction_and_max_%s_gradient_difference'
                             % flow_var] = angle_fish_max_vel_grd
        except Exception as msg:
            print('No nearest neigbours found assigning nans', msg)
            fish_flow_df.loc[ix,
                             'Angle_swim_direction_and_max_%s_gradient_difference'
                             % depth_var] = np.nan
            fish_flow_df.loc[ix,
                             'Angle_swim_direction_and_max_%s_gradient_difference'
                             % flow_var] = np.nan
    print('saving df')
    fish_flow_df.rename(
        columns={'Velocity': 'Fish_swim_velocity_m_per_s',
                 'x_fish': 'Fish_x_coord',
                 'y_fish': 'Fish_y_coord',
                 'index_of_grid_node': 'Index_of_grid_node',
                 'fish_angle': 'Fish_swim_direction_compared_to_x_axis',
                 'flow_angle': 'Flow_direction_compared_to_x_axis',
                 'angle_diff': 'Angle_between_swim_and_flow_direction'},
        inplace=True)
    deltax = fish_flow_df.Fish_x_coord.diff()
    deltay = fish_flow_df.Fish_x_coord.diff()
    fish_flow_df['Time'] = fish_flow_df.index
    fish_flow_df['Time_difference_in_s'] = np.round(
        fish_flow_df.Time.diff() / pd.Timedelta('1s'), 1)
    fish_flow_df['Traveled_distance_in_m'] = calculate_distance_2_points(deltax,
                                                                         deltay)
#         fish_flow_df_final.drop('Time', axis=1, inplace=True)

    cols_new = ['Longitude', 'Latitude', 'Fish_x_coord',
                'Fish_y_coord', 'Time_difference_in_s',
                'Traveled_distance_in_m',
                'Fish_swim_velocity_m_per_s', 'HPE', 'RMSE',
                'Flow_Cat', 'Index_of_grid_node',
                'X_of_grid_node', 'Y_of_grid_node', 'Z_of_grid_node',
                'depth_%s' % flow_val, 'velX_%s' % flow_val,
                'velY_%s' % flow_val, 'velM_%s' % flow_val,
                'Fish_swim_direction_compared_to_x_axis',
                'Flow_direction_compared_to_x_axis',
                'Angle_between_swim_and_flow_direction',
                'Angle_swim_direction_and_max_%s_gradient_difference'
                % depth_var,
                'Angle_swim_direction_and_max_%s_gradient_difference'
                % flow_var]
    # change column names and save df
    fish_flow_df_final = fish_flow_df[cols_new]

    fish_flow_df_final.to_csv(
        os.path.join(out_plots_dir,
                     r'fish_%s_with_flow_data_%s_angles'
                     r'_and_max_gradients.csv'
                     % (fish_nbr, flow_cat)))  # , compression='gzip')

    return fish_flow_df_final
# =============================================================================
#
# =============================================================================


if __name__ == '__main__':

    in_orig_stn_df = read_OrigStn_DF(orig_station_file)

#    in_fish_files_dict = getFiles(main_data_dir, '.csv')  # fish_file[-9:-4]
# in_fish_files_dict = getFiles(points_in_river, '.csv')  #
# fish_file[-25:-20]

#    in_fish_files_dict = getFiles(r'C:\Users\hachem\Desktop\Work_with_Matthias_Schneider\out_plots_abbas\Filtered_data', '.csv')
    in_fish_files_dict = getFiles(r'C:\Users\hachem\Desktop\Work_with_Matthias_Schneider'
                                  r'\out_plots_abbas\df_fish_flow_combined_with_angles',
                                  '.csv')   #
    fish_nbrs = []
    orig_data = []
    rem_data = []
    ratio_data = []

#    plot_flow_fish_values(img_loc, 'Flow_Cat', [20, 30, 40], 'all')
    for fish_type in in_fish_files_dict.keys():
        if fish_type == '2_barbel':
            for fish_file in in_fish_files_dict[fish_type]:
                #             print(fish_file)
                #             print(fish_file)
                # '_all_data_' + fish_file[-22:-17]  #    # fish_file[-32:-27]
                fish_nbr = fish_type + '_' + \
                    fish_file[-47:-42]  # fish_file[-41:-36]

                if fish_nbr == '2_barbel_46861':
                    #             if fish_nbr == '1_grayling_46907':

                    # raise Exception
                    # 'not_considered'  # fish_file[-11:-5]
                    flow_cat = fish_file[-11:-5]  # fish_file[-20:-14]  #

        #                 fish_flow_df = pd.read_csv(fish_file, sep=',', index_col=0,
        #                                            engine='c')
        #                 fish_flow_df.reset_index(level=0, inplace=True)
        #                 fish_flow_df.to_feather(
        #                     os.path.join(out_plots_dir, r'df_fish_flow_combined_with_angles',
        #                                  r'fish_%s_with_flow_data_%s_and_angles.ft'
        #                                  % (fish_nbr, flow_cat)))

            #                 try:
                #                 pass
    # if fish_file ==
    # r'C:\Users\hachem\Desktop\Work_with_Matthias_Schneider\out_plots_abbas\df_fish_flow_combined_with_angles\fish_1_grayling_46872_with_flow_data_and_angles_cat_cat_80_.csv':

                    try:
                        print(fish_file)
                        d = calc_max_gradient_direct(
                            fish_file, flow_cat, fish_nbr)
                # print(d)
    #                 raise Exception
                        plot_difference_in_angle(d, fish_nbr, flow_cat,
                                                 'Angle_swim_direction_and_max_depth_%s_gradient_difference'
                                                 % str(flow_cat[-2:]))
                        plot_difference_in_angle(d, fish_nbr, flow_cat,
                                                 'Angle_swim_direction_and_max_velM_%s_gradient_difference'
                                                 % str(flow_cat[-2:]))
                    except Exception as msg:
                        print(msg)
                        continue
        #                raise Exception
        #                        dd = compare_fish_and_flow_direction(fish_file,
        #                                                             fish_nbr,
        #                                                             flow_cat,
        #                                                             True)
        #                        vel_vls, angle_vls, grdx, grdy = aggregate_values_per_grid_cell(
        #                            delta_x_y, 'Velocity', 'fish_angle')
        #                        plot_agg_grid_vls(grdx, grdy, angle_vls, fish_nbr, 'fish_angle')
        ##                        plot_agg_grid_vls(grdx, grdy, vel_vls, fish_nbr, 'Velocity')
        #                find_diff_fish_and_flow_direction(fish_file, fish_nbr, flow_cat)
#
#                 except Exception as msg:
#                     print(msg)
#                     continue
#                 break
#             break

#            transform_variables_to_uniform_marginals(in_f,
#                                                   'Velocity', 'velM_60')
#
#            df_fish_flow = save_cat_flow_data(flow_data_file,
#                                              simulated_flow_file,
#                                              delta_x_y,
#                                              fish_nbr)

#                plot_heatmapt_fish_loc(delta_x_y, fish_nbr,
#                                        'using_HPE_RMSE_Vel_filters',
#                                        plt_img=True)

    STOP = timeit.default_timer()  # Ending time
    print(('\n\a\a\a Done with everything on %s. Total run time was'
           ' about %0.4f seconds \a\a\a' % (time.asctime(), STOP - START)))
