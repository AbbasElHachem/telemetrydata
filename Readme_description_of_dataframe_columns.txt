Description of each Column:

1. Index: Time of observation of Fish position
2. Longitude: First  coordinate of the observation in coordinates system EPSG:WGS84
3. Latitude: Second coordinates of the observation in coordinates system EPSG:WGS84

4. Fish_x_coord: Transformed longitude to local coordinates system EPSG:UTM32
5. Fish_y_coord: Transformed latitude to local coordinates system EPSG:UTM32

6. Time_difference_in_s: Time difference between each two consecutive observations in seconds: DeltaT = (T2 - T1) 
7. Traveled_distance_in_m: Traveled distance between each two consecutive observations in meters: DeltaXY = SQRT((X2-X1)^2+(Y2-Y1)^2)

8. Fish_swim_velocity_m_per_s: Fish swimming velocity, calculated in m/s as the ratio DeltaXY/DeltaT

9. HPE: VEMCO measure of uncertainty in position
10. RMSE: VEMCO measure of uncertainty in position

11. Flow_Cat: The observed flow category at the time of observation, derived from the river observed discharge
12. Index_of_grid_node: The index of the nearest grid node in the 1m grid used for the Hydraulic model
13. X_of_grid_node: The X coordinate of the nearest grid node in the 1m grid used for the Hydraulic model
14. Y_of_grid_node: The Y coordinate of the nearest grid node in the 1m grid used for the Hydraulic model
15. Z_of_grid_node: The Z coordinate of the nearest grid node in the 1m grid used for the Hydraulic model
16. depth_20: The water level depth of the nearest grid node in the 1m grid used for the Hydraulic model
17. velX_20: The flow velocity vector in the X direction of the nearest grid node in the 1m grid used for the Hydraulic model
18. velY_20: The flow velocity vector in the Y direction of the nearest grid node in the 1m grid used for the Hydraulic model
19. velM_20: The flow velocity magnitude of the nearest grid node in the 1m grid used for the Hydraulic model in m/s

20. Fish_swim_direction_compared_to_x_axis: Fish swimming direction between each two consecutive positions, derived from the coordinates and calculated as compared to the x-axis
21. Flow_direction_compared_to_x_axis: Flow direction at each grid point, derived from the angle between the flow magnitude and the x-axis
22. Angle_between_swim_and_flow_direction: The smallest angle between the Fish swimming direction and the Flow direction, calculated for every position
23. Angle_swim_direction_and_max_depth_20_gradient_difference: Calculated angle between Fish swimming direction and the direction of the maximum depth gradient considering the nearest 4 grid points 
24. Angle_swim_direction_and_max_velM_20_gradient_difference: Calculated angle between Fish swimming direction and the direction of the maximum flow magnitude gradient considering the nearest 4 grid points