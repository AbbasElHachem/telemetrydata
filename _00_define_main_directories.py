# !/usr/bin/env python.
# -*- coding: utf-8 -*-

"""
In this script all the different Path to the data
and output directories are defined.
These are later imported in the corresponding scripts

For the code to run directories are to be defined at first

"""

__author__ = "Abbas El Hachem"
__copyright__ = 'Institut fuer Wasser- und Umweltsystemmodellierung - IWS'
__email__ = "abbas.el-hachem@iws.uni-stuttgart.de"

# =============================================================================

import os

dir_kmz_for_fish_names = (r'E:\Work_with_Matthias_Schneider'
                          r'\2018_11_26_tracks_fish_vemco\kmz')

# dir_kmz_for_fish_names = r'C:\Users\Abbas\Desktop\Work_with_Matthias_Schneider\2018_11_26_tracks_fish_vemco\kmz'
assert os.path.exists(dir_kmz_for_fish_names)

orig_data_dir = (r'E:\Work_with_Matthias_Schneider'
                 r'\2018_11_26_tracks_fish_vemco\csv')
# assert os.path.exists(orig_data_dir)

orig_station_file = (r'E:\Work_with_Matthias_Schneider'
                     r'\2018_11_26_tracks_fish_vemco\stations.csv')
# assert os.path.exists(orig_station_file)

main_data_dir = (r'E:\Work_with_Matthias_Schneider'
                 r'\2018_11_26_tracks_fish_vemco')
# assert os.path.exists(main_data_dir)
# os.chdir(main_data_dir)

# out_plots_dir = r'E:\Work_with_Matthias_Schneider\out_plots_abbas'
out_data_dir = (r'C:\Users\hachem\Desktop'
                r'\Work_with_Matthias_Schneider\out_plots_abbas')

shp_path = (r'E:\Work_with_Matthias_Schneider'
            r'\QGis_abbas\wanted_river_section.shp')
# assert os.path.exists(shp_path)

img_loc = r'E:\Work_with_Matthias_Schneider\GIS\orthoAll_small.jpg'
# assert os.path.exists(img_loc)

asci_grd_file_1m_ = (r'C:\Users\hachem\Desktop\Work_with_Matthias_Schneider'
                     r'\2019_01_18_GridsFromHydraulicModelForIne'
                     r'\altusried_1m_copy.csv')
# assert os.path.exists(asci_grd_file_1m_)

fish_shp_path = (r'C:\Users\hachem\Desktop\Work_with_Matthias_Schneider'
                 r'\QGis_abbas\fish_pass.shp')
# assert os.path.exists(fish_shp_path)

river_shp_path = (r'C:\Users\hachem\Desktop\Work_with_Matthias_Schneider'
                  r'\complere_river_shp\Altusried_Mesh_Boundary.shp')
# assert os.path.exists(river_shp_path)

observed_flow_data = (r"C:\Users\hachem\Desktop\Work_with_Matthias_Schneider"
                      r"\Flow data\q_summe_data_abbas.csv")

simulated_flow_data = (r"C:\Users\hachem\Desktop\Work_with_Matthias_Schneider"
                       r"\2019_01_18_GridsFromHydraulicModelForIne"
                       r"Altusried_hydraulics_Grid1m Copy.csv")
