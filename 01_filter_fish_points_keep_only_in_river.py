# -*- coding: utf-8 -*-
"""
Created on 15-01-2019

@author: EL Hachem Abbas,
Institut fuer Wasser- und Umweltsystemmodellierung - IWS
"""
from __future__ import division

import os
import time
import timeit

from shapely.geometry import shape, Point

import pyproj
import shapefile
import pandas as pd

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
        out_save_dir = os.path.join(out_save_dir, fish_type)
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

def save_all_df_in_river(fish_file, out_data_dir, shp_path):
    ''' intersect points and shapefile river'''

    fish_tag_nbr = fish_file[-9:-4]
    print('working for file, ', fish_tag_nbr)
    df_orig = readDf(fish_file)
    find_fish_in_river(df_orig, shp_path, out_data_dir, fish_tag_nbr)
    print('done saving data for: ', fish_tag_nbr)

    return


# =============================================================================
#
# =============================================================================

if __name__ == '__main__':

    print('\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime())
    START = timeit.default_timer()  # to get the runtime of the program

    dir_kmz_for_fish_names = (r'E:\Work_with_Matthias_Schneider'
                              r'\2018_11_26_tracks_fish_vemco\kmz')
    assert os.path.exists(dir_kmz_for_fish_names)

    orig_station_file = (r'E:\Work_with_Matthias_Schneider'
                         r'\2018_11_26_tracks_fish_vemco\stations.csv')
    assert os.path.exists(orig_station_file)

    main_data_dir = (r'E:\Work_with_Matthias_Schneider'
                     r'\2018_11_26_tracks_fish_vemco')
    assert os.path.exists(main_data_dir)
    os.chdir(main_data_dir)

    # out_plots_dir = r'E:\Work_with_Matthias_Schneider\out_plots_abbas'
    out_data_dir = (r'C:\Users\hachem\Desktop'
                    r'\Work_with_Matthias_Schneider\out_plots_abbas')

    shp_path = (r'E:\Work_with_Matthias_Schneider'
                r'\QGis_abbas\wanted_river_section.shp')
    assert os.path.exists(shp_path)

    # def epsg wgs84 and utm32
    wgs82 = "+init=EPSG:4326"
    utm32 = "+init=EPSG:32632"

    # start filtering the data, keep only in river
    in_orig_stn_df = read_OrigStn_DF(orig_station_file)

    in_fish_files_dict = getFiles(main_data_dir, '.csv')
    # call function
#    multi_proc_shp_inter(in_fish_files_dict)
    for fish_type in in_fish_files_dict.keys():

        for fish_file in in_fish_files_dict[fish_type]:
            save_all_df_in_river(fish_file, out_data_dir, shp_path)
            break
        break
