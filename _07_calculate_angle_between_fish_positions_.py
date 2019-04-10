# !/usr/bin/env python.
# -*- coding: utf-8 -*-

"""
Calculate angle between consecutive fish positions

Parameters
----------
Filtered observed fish dataframe, incluiding longitude and latitude
for point cordinates

Returns
-------
a dataframe with a column consindering the angle between each two positions
"""

__author__ = "Abbas El Hachem"
__copyright__ = 'Institut fuer Wasser- und Umweltsystemmodellierung - IWS'
__email__ = "abbas.el-hachem@iws.uni-stuttgart.de"

# ===================================================

from pathlib import Path

from _01_filter_fish_points_keep_only_in_river import getFiles

import os
import timeit
import time

import numpy as np
import pandas as pd

main_dir = Path(os.getcwd())
os.chdir(main_dir)

#==============================================================================
# # def all directories and all required parameters
#==============================================================================
dir_kmz_for_fish_names = (r'E:\Work_with_Matthias_Schneider'
                          r'\2018_11_26_tracks_fish_vemco\kmz')
assert os.path.exists(dir_kmz_for_fish_names)


filtered_data_dir = (r'C:\Users\hachem\Desktop'
                     r'\Work_with_Matthias_Schneider'
                     r'\out_plots_abbas\Filtered_df_HPE_RMSE_VEL')
assert os.path.exists(filtered_data_dir)

out_save_dir = (r'C:\Users\hachem\Desktop\Work_with_Matthias_Schneider'
                r'\out_plots_abbas\'Df_filtered_with_angles')
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
if __name__ == '__main__':

    print('**** Started on %s ****\n' % time.asctime())
    START = timeit.default_timer()  # to get the runtime of the program

    in_filtered_fish_files_dict = getFiles(filtered_data_dir, '.csv',
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
