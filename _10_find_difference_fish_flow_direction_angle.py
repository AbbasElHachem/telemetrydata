import os
import pandas as pd
import numpy as np


from _00_define_main_directories import out_data_dir
from _07_calculate_angle_between_fish_positions_ import (
    calculate_angle_between_two_positions)


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


def find_diff_fish_and_flow_direction(fish_file, fish_type_nbr, flow_cat, out_plots_dir):
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
