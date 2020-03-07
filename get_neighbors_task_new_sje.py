
# coding: utf-8

# In[32]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial


# In[4]:


df_file = r"C:\Users\hachem\Downloads\Altus_hydro_Fish.csv"


# In[5]:


in_df = pd.read_csv(df_file, sep=',', index_col=0)


# In[6]:


in_df.head()


# In[9]:


# create columns
depth_cols = ['d_%d' %i for i in range(1, 9)]
velocity_cols = ['v_%d' %i for i in range(1, 9)]

assert velocity_cols[0] in in_df.columns


# In[11]:


# get coords as np arrays
x_coords, y_coords = in_df.x.values.ravel(), in_df.y.values.ravel()


# In[19]:


# create a tree from coordinates
coords_tuples = np.array([(x, y) for x, y in zip(x_coords, y_coords)])
points_tree = spatial.cKDTree(coords_tuples)


# In[48]:


# test random point
x0, y0 = x_coords[100], y_coords[100]
d0 = in_df.loc[:, 'd_1'].values[0]


# In[49]:


# get distances, indices of 9 neighbours (point included)
distances, indices = points_tree.query([x0, y0], k=9)


# In[56]:


distances, indices


# In[62]:


# get coordinates of neighbors
coords_nearest_nbr = coords_tuples[indices[:]]

# get df related to neighbors
df_neighbors = in_df.iloc[indices,:]

x_coords_neighbrs = df_neighbors.x.values.ravel()
y_coords_neighbrs = df_neighbors.y.values.ravel()


# In[63]:


# plot to check if it works
plt.ioff()
plt.scatter(x0, y0, c='r')
plt.scatter(x_coords_neighbrs, y_coords_neighbrs, c='b', alpha=0.25)
plt.show()


# In[ ]:


# start iterating through ids and columns
for point_id in in_df.index[:1000]:  # test first 1000 points
    print('id is ', point_id)
    x_val = in_df.loc[point_id, 'x']
    y_val = in_df.loc[point_id, 'y']
    
    # find neighbors
    distances, indices = points_tree.query([x_val, y_val], k=9)
    
    # excluding center point itself
    df_neighbors = in_df.iloc[indices[1:],:]
    
    x_coords_neighbors = df_neighbors.x.values.ravel()
    y_coords_neighbrs = df_neighbors.y.values.ravel()
    ids_neighbors = df_neighbors.index
    # start going through depth columns
    for depth_col, vel_col in zip(depth_cols, velocity_cols):
        
        depth_neighbors = df_neighbors.loc[ids_neighbors, depth_col]
        ratio_depth_distance = (depth_neighbors / distances[1:])
        max_ratio_depth_distance = ratio_depth_distance.max()
        
        vel_neighbors = df_neighbors.loc[ids_neighbors, vel_col]
        ratio_vel_distance = (vel_neighbors / distances[1:])
        max_ratio_vel_distance = ratio_vel_distance.max()
        
        in_df.loc[point_id,
                  'max_ratio_depth_distance_%s'
                  % depth_col] = max_ratio_depth_distance
        in_df.loc[point_id,
                  'max_ratio_velocity_distance_%s'
                  % vel_col] = max_ratio_vel_distance
        
in_df.to_csv(r"C:\Users\hachem\Downloads\Altus_hydro_Fish_modified_Abbas.csv", sep=';')   


