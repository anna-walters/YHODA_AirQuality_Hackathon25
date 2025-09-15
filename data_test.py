
import pandas as pd
import matplotlib.pyplot as plt
import pathlib as pl
import cartopy 

import cartopy.crs as ccrs
from cartopy.io.img_tiles import OSM

#REFERENCES:
#
# https://cartopy.readthedocs.io/stable/gallery/miscellanea/tube_stations.html#sphx-glr-gallery-miscellanea-tube-stations-py

# is x + y northing & easting?


def plot_site_locations(dataframe):
    imagery = OSM()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=imagery.crs)
    #ax.set_extent([-0.14, -0.1, 51.495, 51.515], ccrs.PlateCarree())

    # Add the imagery to the map.
    ax.add_image(imagery, 14)

    # Plot the locations twice, first with the red concentric circles,
    # then with the blue rectangle.
    xs, ys = dataframe['x'], dataframe['y']
    ax.plot(xs, ys, transform=ccrs.OSGB(approx=False),
            marker='x', color='red', markersize=9, linestyle='')

    ax.set_title('Site Locations')
    plt.show()


def load_data():
    file_path = pl.Path('datasets') / 'air_quality_monitoring_results_2003_to_2024_1.xlsx'
    dataframe = pd.read_excel(file_path, header=1) 
    dataframe = dataframe.reset_index(drop=True)
    return dataframe


if __name__ == "__main__":
    dataframe = load_data()
    plot_site_locations(dataframe)