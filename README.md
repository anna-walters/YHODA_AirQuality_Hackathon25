
# YHODA / SYSC Hackathon 2025 - Challenge 2 - Air Pollution in Sheffield.

## Packages Used:
- matplotlib
- numpy
- pandas
- cartopy
- pathlib

## Historical Pollutant Data Sources
Ozone, Carbon Monoxide, Sulfur Dioxide, PM2.5, PM10 - https://uk-air.defra.gov.uk/data/pcm-data

Nitrogen Dioxide - Map of Air Quality Monitoring Sites in Sheffield, 2003-2004 Associated Data

Several years of these datasets were compiled into a single dataset per pollutant, making up the compiled_{POLLUTANT}_data.csv files.

## Predictive Data Sources

Nitrogen Dioxide (2021 & 2040) - https://uk-air.defra.gov.uk/data/laqm-background-maps?year=2021


## get_data.py

This file contains functions for manipulating the raw data, e.g compiling it into a single csv file.

## solution_test.py

This file contains plots based on splitting the data into "inside" and "outside" of the clean air zone area. 
Whether sites were inside or outside 0was determined manually. This code is ultimately unused in our presentation due to there being very few datapoints where the zone is active.

## no2.py
This file contains several functions for plotting NO2 data.

plot_site_locations, plot_site_locations_colour_by_ward, plot_sites_where_over_2005_limit use Map of Air Quality Monitoring Sites in Sheffield, 2003-2004 Associated Data

plot_1km_grid_coloured_by_no2 uses the predictive data from DEFRA.

# REFERENCES:
https://cartopy.readthedocs.io/stable/gallery/miscellanea/tube_stations.html#sphx-glr-gallery-miscellanea-tube-stations-py


