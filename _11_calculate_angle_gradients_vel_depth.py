from __future__ import division

import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import pickle
from matplotlib.ticker import FormatStrFormatter
from scipy import spatial

from _00_define_main_directories import dir_kmz_for_fish_names
from _01_filter_fish_points_keep_only_in_river import getFiles
from _02_filter_fish_data_based_on_HPE_Vel_RMSE import calculate_distance_2_points

fontsize, labelsize = 10, 8


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
    x_grid = fish_flow_df.X_of_grid_node.values
    y_grid = fish_flow_df.Y_of_grid_node.values
    depth_var = 'depth_%s' % flow_val
    flow_var = 'velM_%s' % flow_val

    list_of_coords = [(x, y) for x, y in zip(x_grid, y_grid)]
    tree = spatial.cKDTree(list_of_coords)
    print('calculting for', fish_flow_file)
    for ix, x0, y0 in zip(fish_flow_df.index, x_grid, y_grid):
        print(ix, x0, y0)
        depth_point0 = fish_flow_df.loc[ix, depth_var]
        flow_vel_point0 = fish_flow_df.loc[ix, flow_var]

        distance, indices = tree.query((x0, y0), k=5)
        diff_in_grds_lst = []
        diff_in_vel_lst = []
        for indice in indices:

            point_i_depth = fish_flow_df.iloc[indice, :][depth_var]
            point_i_flow_mag = fish_flow_df.iloc[indice, :][flow_var]

            diff_in_grds_lst.append(np.abs(depth_point0 - point_i_depth))
            diff_in_vel_lst.append(np.abs(flow_vel_point0 - point_i_flow_mag))

        (_, depth_point_id) = (np.max(diff_in_grds_lst),
                               indices[np.argmax(diff_in_grds_lst)])
        (_, vel_point_id) = (np.max(diff_in_vel_lst),
                             indices[np.argmax(diff_in_vel_lst)])

        (x_d, y_d) = (fish_flow_df.iloc[depth_point_id, :].x_fish, # Fish_x_coord
                      fish_flow_df.iloc[depth_point_id, :].y_fish) # Fish_y_coord
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
    fish_flow_df.drop('Time', axis=1, inplace=True)

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
    fish_flow_df = fish_flow_df[cols_new]

    df_name = os.path.join(out_plots_dir,
                           r'fish_%s_with_flow_data_%s_angles'
                           r'_and_max_gradients.npy'
                           % (fish_nbr, flow_cat))
    np.save(open(df_name, 'w'), fish_flow_df.values)
    meta = fish_flow_df.index, fish_flow_df.columns
    s = pickle.dumps(meta)
    s = s.encode('string_escape')
    with open(df_name, 'a') as f:
        f.seek(0, 2)
        f.write(s)
#    fish_flow_df.to_csv(
#        os.path.join(out_plots_dir,
#                     r'fish_%s_with_flow_data_%s_angles'
#                     r'_and_max_gradients.csv'
#                     % (fish_nbr, flow_cat)))  # , compression='gzip')

    return  # fish_flow_df
# =============================================================================
# https://metarabbit.wordpress.com/2013/12/10/how-to-save-load-large-pandas-dataframes/
# =============================================================================


if __name__ == '__main__':

    in_fish_files_dict = getFiles(r'C:\Users\Abbas\Desktop\Work_with_Matthias_Schneider'
                                  r'\out_plots_abbas\df_fish_flow_combined_with_angles',
                                  '.csv', dir_kmz_for_fish_names)

    out_plots_dir = r'C:\Users\Abbas\Desktop\Work_with_Matthias_Schneider\out_plots_abbas'

    for fish_type in in_fish_files_dict.keys():

        for fish_file in in_fish_files_dict[fish_type]:
            print(fish_file)

            fish_nbr = fish_type + '_' + fish_file[-47:-42]

            flow_cat = fish_file[-11:-5]

            try:
                d = calc_max_gradient_direct(fish_file, flow_cat, fish_nbr)

            except Exception as msg:
                print(msg)
                continue
