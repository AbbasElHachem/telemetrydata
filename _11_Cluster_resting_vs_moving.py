
# **Clustering based on distance covered**
#
# Fish positions (of one fish) are first resampled over e.g. 5 minutes
# (i.e. take the average position in a bin of 5 minutes).
# Than the distance covered between one (or two) previous and subsequent
# (resampled) point(s) is calculated. This distance is used to seperate the groups:
# if the distance covered is below a certain threshold, the fish is
# resting, otherwise it is moving.

# # Read-ins

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
# INTERIMPATH = './'

INTERIMPATH = r'C:\Users\hachem\Desktop\Work_with_Matthias_Schneider\Altusried_XY_plots'
# INTERIMPATH = r'C:\Users\Abbas\Desktop\Work_with_Matthias_Schneider\Altusried_XY_plots'
assert os.path.exists(INTERIMPATH)
# In[2]:
plot_results = False


# def distance(df, window):
#     if window % 2 > 0:
#         print('Only even windows!')
#         res = pd.NaT
#     else:
#         res = np.sqrt((df.x_new.shift(int(window / 2)) - df.x_new.shift(-int(window / 2)))**2 +
#                       (df.y_new.shift(int(window / 2)) - df.y_new.shift(-int(window / 2)))**2)
#     return res

def distance(df, xcol, ycol, window):
    if window % 2 > 0:
        print('Only even windows!')
        res = pd.NaT
    else:

        res = np.sqrt((df[xcol].shift(int(window / 2)) - df[xcol].shift(-int(window / 2)))**2 +
                      (df[ycol].shift(int(window / 2)) - df[ycol].shift(-int(window / 2)))**2)

    return res

# In[116]:


fish_df = pd.read_csv(INTERIMPATH + r'\fish_2_barbel_46838_with_flow_data_cat_10_angles_and_max_gradients.csv',
                      sep=',', index_col=0, parse_dates=True)
# graylings_df = pd.read_pickle(INTERIMPATH + r'\graylings_pos_par.pkl')
# barbels_df = pd.read_pickle(INTERIMPATH + 'r\barbels_pos_par.pkl')

# In[61]:

FL_tracks_grayling = pd.read_pickle(INTERIMPATH + r'\FL_tracks_grayling.pkl')
FL_tracks_barbel = pd.read_pickle(INTERIMPATH + r'\FL_tracks_barbel.pkl')

# In[42]:

# barbels_df.ID.unique()

# In[125]:

# example fish: 46906


# ID = 46863
ID = 46838
# fish_df = graylings_df[graylings_df.ID == str(ID)].copy()
# fish_df2 = barbels_df[barbels_df.ID == str(ID)].copy()


# ID = 46863

# fish_df2 = graylings_df[graylings_df.ID == str(ID)].copy()
# fish_df = barbels_df[barbels_df.ID==str(ID)].copy()


# # Function for clustering

# In[55]:


def resting_vs_moving(fish_df, sample_bin='5min', window=4, distance_threshold=10, min_elements=5):
    """
    fish_df : df, containing positions of one fish
    sample_bin : Str, time bin to resample, e.g. '5min'
    window : Int (even number), window over which to calculate the distance covered (the point itself is at the center of this window)
    distance_threshold : Int, maximum distance to be covered in the window, to define the fish as 'resting' at that point
    min_elements : Int, min nb of elements in a segment to accept it

    """

    # resample to calculate an average position in 5 minutes time
    fish_df['Time'] = fish_df.index
    fish_resampled = fish_df.set_index('Time').resample(
        sample_bin).mean().dropna().reset_index(drop=False)
#     fish_resampled = fish_df.index.resample(
#         sample_bin).mean().dropna().reset_index(drop=False)
    # calculate the distance between previous and next resampled point => if
    # more than 10 m => gap
    gaps = distance(fish_resampled.dropna(), 'Fish_x_coord', 'Fish_y_coord',
                    window=window).abs() > distance_threshold  # m
    # create a different group each time there is a gap
    groups = gaps.cumsum()
    # groups.diff() => where this is 1, fish was moving, where this is 0, fish
    # was resting (at least within 5m over 10 minutes)
    fish_resampled_grouped = fish_resampled.groupby(groups.diff().fillna(1.0))
    fish_resampled['group'] = groups.diff().fillna(1.0)

    # fill first na with 0 (because previous element can not be compared, so
    # put this in one group)
    fish_resampled['group_diff'] = fish_resampled.group.diff().fillna(0.0)
    # make a new segment each time that group changes from 1 to 0 or invers
    fish_resampled['segment'] = fish_resampled['group_diff'].abs(
    ).cumsum().astype(int)
    segments = fish_resampled.groupby(by='segment')
    segment_lengths = segments.apply(lambda seg: len(seg))
    # Each segment that contains at least 5 elements is accepted
    # Begin and ends of segments form boundaries of final clusters
    keys_to_keep = segment_lengths[segment_lengths >= min_elements].keys()
    segments_list = [segments.get_group(key) for key in keys_to_keep]
    segments_df = pd.concat(segments_list)

    # Begin and ends of segments form boundaries of final clusters
    segment_summary = pd.DataFrame(index=keys_to_keep, columns=[
                                   'begin', 'end', 'group'])

    segment_summary['begin'] = [segments.get_group(
        key).Time.min() for key in keys_to_keep]
    # add sample_bin mins to end time, because resampling time is bottom time
    # of next sample_bin mins (-1 second to avoid taking next minute as well)
    segment_summary['end'] = [segments.get_group(key).Time.max(
    ) + pd.Timedelta(sample_bin) - pd.Timedelta('1s') for key in keys_to_keep]
    # 0 = resting, 1 = moving
    segment_summary['group'] = [list(segments.get_group(key).group)[
        0] for key in keys_to_keep]

    resting_segments = segment_summary[segment_summary.group == 0]

    # put "group" column on 1 (moving) => resting segments will be put on 0
    fish_df['group'] = 1
    fish_df = fish_df.set_index('Time')

    for i in resting_segments.index:
        fish_df.loc[segment_summary.loc[i].begin:segment_summary.loc[i].end, 'group'] = 0

    fish_df = fish_df.reset_index(drop=False)

    return fish_df, segments_df

# In[123]:


fish_df, segments_df = resting_vs_moving(
    fish_df, sample_bin='5min', window=4, distance_threshold=10, min_elements=5)
moving = fish_df[fish_df.group == 1.0]
resting = fish_df[fish_df.group == 0.0]
moving_seg = segments_df[segments_df.group == 1.0]
resting_seg = segments_df[segments_df.group == 0.0]

fish_df.to_feather(
    os.path.join(r'C:\Users\hachem\Desktop\Work_with_Matthias_Schneider\out_plots_abbas', r'df_fish_flow_combined_with_angles',
                 r'fish_barbel_%s_with_flow_data_%s_and_angles_and_behaviour.ft'
                 % (ID, '10')))
# # Plot results

# In[124]:

if plot_results:
    fig, (ax, ax2, ax3) = plt.subplots(nrows=3, sharex=True, figsize=(8, 9))

    ax.plot(resting.Time, resting.Fish_x_coord, marker='.',
            lw=0, c='green', label='resting')
    ax.plot(moving.Time, moving.Fish_x_coord, marker='.',
            lw=0, c='red', label='moving')
    ax.legend()
    ax.set_title('Movement in x-direction of original data')
    # ax.set_xlim(('2018-06-05 06', '2018-06-07 12'))

    ax2.plot(resting.Time, resting.Fish_y_coord, marker='.',
             lw=0, c='green', label='resting')
    ax2.plot(moving.Time, moving.Fish_y_coord, marker='.',
             lw=0, c='red', label='moving')
    ax2.legend()
    ax2.set_title('Movement in y-direction of original data')

    ax3.plot(resting_seg.Time, resting_seg.Fish_x_coord,
             marker='.', lw=0, c='green', label='resting')
    ax3.plot(moving_seg.Time, moving_seg.Fish_x_coord,
             marker='.', lw=0, c='red', label='moving')
    ax3.legend()
    ax3.set_title('Movement in x-direction for only accepted segments')

    fig.autofmt_xdate()

    # Some test periods to check (for 46906)
    # ax.set_xlim(('2018-05-09 03', '2018-05-10 00'))
    # ax.set_xlim(('2018-04-04 12:20', '2018-04-05'))
    # ax.set_xlim(('2018-04-06 02','2018-04-06 04'))
    # ax.set_xlim(('2018-05-07 12','2018-05-10 00'))

# # Check migration tracks to fishladder

# In[128]:

fishladder_tracks = FL_tracks_barbel[FL_tracks_barbel.ID == ID]
# fishladder_tracks = FL_tracks_grayling[FL_tracks_grayling.ID == ID]

# In[127]:

# check if the fishladder tracks (if any) are part of the data classified
# as "moving"
for check_time in list(fishladder_tracks.time_in):
    if len(moving.set_index('Time')[pd.to_datetime(check_time) - pd.Timedelta('30min'):check_time]) > 5:
        print('Fishladder track ending at ' +
              str(check_time.round('1min')) + ' is in moving data.')


fishladder_tracks = FL_tracks_barbel[FL_tracks_barbel.ID == ID]
# fishladder_tracks = FL_tracks_grayling[FL_tracks_grayling.ID == ID]

# # Check some parameters for both groups


# In[129]:
#
# fig, ax = plt.subplots()
# ax.boxplot([resting.turning_angle.dropna(),
#             moving.turning_angle.dropna()], positions=[1, 2])
# ax.set_xticks([1, 2])
# ax.set_xticklabels(['resting', 'moving'])
# ax.set_title('Turning angles (angle between previous and next position)')
# ax.set_ylim(-5, 185)
# ax.set_yticks(np.arange(0, 181, 30))
# ax.set_ylabel('angle °')
#
# # **Resting positions tend to have more <90° turning angles. Moving positions have more > 90° angles, indicating movement in a certain direction, following a line or curve.**
#
#
# # check if the fishladder tracks (if any) are part of the data classified
# # as "moving"
# for check_time in list(fishladder_tracks.time_in):
#     if len(moving.set_index('Time')[pd.to_datetime(check_time) - pd.Timedelta('30min'):check_time]) > 10:
#         print('Fishladder track ending at ' +
#               str(check_time.round('1min')) + ' is in moving data.')
#
# # In[130]:
#
#
# fig, ax = plt.subplots()
# ax.boxplot([resting.flow_fish_angle.dropna(),
#             moving.flow_fish_angle.dropna()], positions=[1, 2])
# ax.set_xticks([1, 2])
# ax.set_xticklabels(['resting', 'moving'])
# ax.set_title('Angle between flow and fish velocity')
# ax.set_ylim(-5, 185)
# ax.set_yticks(np.arange(0, 181, 30))
# ax.set_ylabel('angle °')
#
# # **Resting positions tend to have angles versus the flow in all possible directions, corresponding to a boxplot of random distribution (centred around the mean angle of 90°). Moving positions have more directed angles, with or against the flow.**
#
# # In[131]:
#
# fig, ax = plt.subplots()
# ax.boxplot([resting.Velocity.dropna(),
#             moving.Velocity.dropna()], positions=[1, 2])
# ax.set_xticks([1, 2])
# ax.set_xticklabels(['resting', 'moving'])
# ax.set_title('Fish velocity')
# ax.set_ylabel('velocity (m/s)')
#
# # **Moving positions tend to have higher velocities.**
#
# # In[132]:
#
# fig, ax = plt.subplots()
# ax.boxplot([resting.TimeOfDay_float.dropna(),
#             moving.TimeOfDay_float.dropna()], positions=[1, 2])
# ax.set_xticks([1, 2])
# ax.set_xticklabels(['resting', 'moving'])
# ax.set_title('Time of day')
# ax.set_ylabel('Time of day (hours)')
# ax.set_ylim(-1, 25)
# ax.set_yticks(np.arange(0, 25, 6))
#
# plt.show()
