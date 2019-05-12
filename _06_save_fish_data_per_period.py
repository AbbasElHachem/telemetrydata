# !/usr/bin/env python.
# -*- coding: utf-8 -*-

"""
Script used to divide Fish dataframe into three predefined periods

Periods are assumed to be related to behavior
Use the second function to save the dataframe per period
"""

__author__ = "Abbas El Hachem"
__copyright__ = 'Institut fuer Wasser- und Umweltsystemmodellierung - IWS'
__email__ = "abbas.el-hachem@iws.uni-stuttgart.de"

# =============================================================================

import os
import timeit
import time

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

#==============================================================================
#
#==============================================================================


def select_df_within_period(df, start, end):
    ''' a function to select df between two dates'''
    mask = (df.index > start) & (df.index <= end)
    df_period = df.loc[mask]
    return df_period

#==============================================================================
#
#==============================================================================


def save_fish_per_period(df_fish, fish_nbr,
                         periods_dict_names_dates,
                         out_plots_dir):
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


if __name__ == '__main__':

    print('**** Started on %s ****\n' % time.asctime())
    START = timeit.default_timer()  # to get the runtime of the program

    STOP = timeit.default_timer()  # Ending time
    print(('\n****Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ***' % (time.asctime(), STOP - START)))
