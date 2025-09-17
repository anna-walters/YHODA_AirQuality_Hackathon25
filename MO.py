
"""import pandas as pd
import matplotlib.pyplot as plt
import pathlib as pl

import cartopy.crs as ccrs
from cartopy.io.img_tiles import OSM
import numpy as np


def plot_1km_grid_coloured_by_no2(dataframe, no2_col, year):
    """
    #Plots 1km grid boxes around each (x, y) location, coloured by total NO2.
    #Assumes dataframe has columns: 'x', 'y', 'total NO2'
"""
    imagery = OSM()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=imagery.crs)
    ax.add_image(imagery, 5)

    xs, ys, no2 = dataframe['x'], dataframe['y'], dataframe[no2_col].astype(float)

    # Create 1km grid boxes centered on each (x, y)
    for xi, yi, n in zip(xs, ys, no2):
        print( xi, yi, n)
        # Each box is 1km x 1km (500m offset from center)
        rect = plt.Rectangle(
            (xi - 500, yi - 500), 1000, 1000,
            linewidth=0.5,
            edgecolor='black',
            facecolor= 'grey' if n > 40 else 'green',
            alpha=0.7,
            transform=ccrs.OSGB(approx=False)
        )
            
        ax.add_patch(rect)


    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=40))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, orientation='vertical', label='Total NO₂')

    ax.set_title(f'1km Grid Boxes Coloured by Total MO - {year}')
    plt.show()"""



import pandas as pd
import matplotlib.pyplot as plt
import pathlib as pl
import numpy as np

def trend_over_time(merged_dataframe):

    year_cols = [f"com8hr{col}" for col in range(2005, 2011)]
    
    xs = list(range(2005, 2011))
    ys_matrix = merged_dataframe[year_cols].astype(float).values
    for ys in ys_matrix:
        plt.plot(xs, ys)

    plt.hlines(y=10, xmin=2005, xmax=2010, colors='r', linestyles='dashed', label='10 µg/m³ Limit')
        
    plt.xlabel('Year')
    plt.ylabel('Carbon Monoxide Levels')
    plt.title('Carbon Monoxide Levels Over Time by Location')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv(pl.Path("COMPILED_DATA")/"compiled_MO_data.csv")
    df.replace('MISSING', np.nan, inplace=True)
    trend_over_time(df)