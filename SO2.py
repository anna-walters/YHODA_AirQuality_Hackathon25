import pandas as pd
import matplotlib.pyplot as plt
import pathlib as pl

import cartopy.crs as ccrs
from cartopy.io.img_tiles import OSM
import numpy as np


def load_data():
    file_path = pl.Path('datasets') / "SO2Data" / "mapso22023.csv"
    dataframe = pd.read_csv(file_path, header = 5)
    return dataframe

def compile_data():
    # Compile data from multiple years into a single DataFrame
    all_data = []
    for year in range(2015, 2024):
        file_path = pl.Path("SO2Data") / f"mapso2{year}.csv"
        df = pd.read_csv(file_path, header=5)
        df = df[['x', 'y', f'so2{year}']]
        all_data.append(df)
    
    merged_dataframe = all_data[0]
    for dataframe in all_data[1:]:
        merged_dataframe = pd.merge(merged_dataframe, dataframe, on=['x', 'y'], how = 'inner')
    
    merged_dataframe.to_csv(pl.Path("SO2Data") / "compiled_so2_data.csv", index=False)
    print(merged_dataframe)

    # Merge all yearly data on 'Site name', 'x', 'y', and 'Council ward'


def trend_over_time(merged_dataframe):
    # Find year columns (assuming they are integers from 2015 to 2024)
    year_cols = [f"so2{col}" for col in range(2015, 2024)]
    # plot a line graph with so2 over time for each (x, y) location
    xs = list(range(2015, 2024))
    ys_matrix = merged_dataframe[year_cols].astype(float).values
    for ys in ys_matrix:
        plt.plot(xs, ys)

    plt.hlines(y=40, xmin=2015, xmax=2023, colors='r', linestyles='dashed', label='40 µg/m³ Limit')
        
    plt.xlabel('Year')
    plt.ylabel('Sulfur Dioxide Levels')
    plt.title('Sulfur Dioxide Levels Over Time by Location')
    plt.legend()
    plt.show()

def plot_1km_grid_coloured_by_no2(dataframe, no2_col, year):
    """
    Plots 1km grid boxes around each (x, y) location, coloured by total NO2.
    Assumes dataframe has columns: 'x', 'y', 'total NO2'
    """
    imagery = OSM()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=imagery.crs)
    ax.add_image(imagery, 14)

    xs, ys, no2 = dataframe['x'], dataframe['y'], dataframe[no2_col]

    # Create 1km grid boxes centered on each (x, y)
    for xi, yi, n in zip(xs, ys, no2):
        # Each box is 1km x 1km (500m offset from center)
        rect = plt.Rectangle(
            (xi - 500, yi - 500), 1000, 1000,
            linewidth=0.5,
            edgecolor='black',
            facecolor=plt.cm.viridis((n - no2.min()) / (no2.max() - no2.min())),
            alpha=0.7,
            transform=ccrs.OSGB(approx=False)
        )
        ax.add_patch(rect)


    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=40))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', label='Total SO₂')

    ax.set_title(f'1km Grid Boxes Coloured by Total SO₂ - {year}')
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv(pl.Path("COMPILED_DATA")/"compiled_so2_data.csv")
    df.replace('MISSING', np.nan, inplace=True)
    trend_over_time(df)
    #plot_1km_grid_coloured_by_no2(df, 'so22023', 2023)