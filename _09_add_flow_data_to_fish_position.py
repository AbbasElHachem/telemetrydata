
'''
A function to read the observed flow data and
observed fish locations, find for every position,
based on time of measure the corresponding observed
flow and categorize the output based on different flow
categories. Save the resulted dataframes

'''
import os
import math
import pandas as pd

from _00_define_main_directories import (observed_flow_data,
                                         simulated_flow_data,
                                         out_data_dir)
from _02_filter_fish_data_based_on_HPE_Vel_RMSE import (convert_coords_fr_wgs84_to_utm32_,
                                                        wgs82,
                                                        utm32)
from _06_save_fish_data_per_period import select_df_within_period

# =============================================================================
#
# =============================================================================


def save_cat_flow_data(obsv_flow_file,
                       simulated_flow_file,
                       df_fish,
                       fish_type_nbr,
                       out_plots_dir):
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
