
import pandas as pd
import matplotlib.pyplot as plt
import pathlib as pl

import cartopy.crs as ccrs
from cartopy.io.img_tiles import OSM
import numpy as np

#REFERENCES:
#
# https://cartopy.readthedocs.io/stable/gallery/miscellanea/tube_stations.html#sphx-glr-gallery-miscellanea-tube-stations-py


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

def plot_site_locations_colour_by_ward(dataframe):
    imagery = OSM()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=imagery.crs)

    # Add the imagery to the map.
    ax.add_image(imagery, 14)

    xs, ys = dataframe['x'], dataframe['y']
    councils = dataframe['Council ward'].unique()
    colors = plt.cm.get_cmap('tab20', len(councils))

    for i, council in enumerate(councils):
        council_data = dataframe[dataframe['Council ward'] == council]
        ax.plot(council_data['x'], council_data['y'], transform=ccrs.OSGB(approx=False),
                marker='o', color=colors(i), markersize=6, linestyle='', label=council)

    ax.set_title('Site Locations by Council Ward')
    fig.legend(loc='upper right', fontsize='small', markerscale=1)
    plt.show()

def plot_ward_averages(dataframe):
    """
    Plots average NO2 levels by council ward for the most recent year (2024).
    """
    # Calculate average NO2 by council ward for 2024

    ward_dataframe = pd.DataFrame(columns = ["Year"] + [ward for ward in dataframe["Council ward"].unique()])
    
    ward_dataframe["Year"] = range(2005, 2025)

    for ward in dataframe["Council ward"].unique():
        ward_data = dataframe[dataframe["Council ward"] == ward]
        for year in range(2005, 2025):
            ward_dataframe.loc[ward_dataframe["Year"] == year, ward] = ward_data[year].mean()

    ward_change = {"Ward": [], "Initial": [], "Final": [], "Change": [], "Percentage Change": []}
    for ward in dataframe["Council ward"].unique():
        initial = ward_dataframe[ward][ward_dataframe[ward].first_valid_index()]
        final = ward_dataframe[ward][ward_dataframe[ward].last_valid_index()]
        change = final - initial
        ward_change["Ward"].append(ward)
        ward_change["Initial"].append(initial)
        ward_change["Final"].append(final)
        ward_change["Change"].append(change)
        ward_change["Percentage Change"].append((change / initial) * 100 if initial else np.nan)

    print(ward_dataframe)
    print(ward_change)

    # Bar chart of changes
    fig, ax = plt.subplots(figsize=(10, 6))
    ax2 = ax.twinx()
    ax.bar(ward_change["Ward"], ward_change["Change"], color='skyblue')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylabel("Change in Mean NO₂ (µg/m³)")
    ax.set_title("Change in Mean NO₂ by Council Ward (2005-2024)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ward_change["Ward"], ward_change["Percentage Change"], color='black', linewidth=0.8)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_ylabel("Percentage Change in Mean NO₂ (%)")
    ax.set_title("Percentage Change in Mean NO₂ by Council Ward (2005-2024)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def plot_sites_where_over_2005_limit(dataframe):
    """
    Plots sites where any year exceeds the 2005 WHO annual mean guideline of 40 µg/m³.
    """
    # Identify sites exceeding the limit in any year
    exceed_sites = dataframe[dataframe[2024] > 40]

    imagery = OSM()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=imagery.crs)
    ax.add_image(imagery, 14)

    xs, ys = exceed_sites['x'], exceed_sites['y']
    ax.plot(xs, ys, transform=ccrs.OSGB(approx=False),
            marker='o', color='red', markersize=6, linestyle='', label='Exceeds 40 µg/m³')

    ax.set_title('Sites Exceeding 2005 WHO NO₂ Limit in 2024')
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
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', label='Total NO₂')

    ax.set_title(f'1km Grid Boxes Coloured by Total NO₂ - {year}')
    plt.show()

def plot_no2_over_time_by_ward(dataframe):
    """
    Plots NO2 levels over time for each site in the dataframe.
    """
    # Melt the dataframe to long format: one row per site/year
    year_cols = [int(col) for col in dataframe.columns.astype(str) if col.startswith("2")]
    df = pd.DataFrame(columns=["Year", "Site name", "Council ward", "Mean NO2"])
    for year in year_cols:
        temp_df = dataframe[["Site name", "Council ward", year]].copy()
        temp_df = temp_df.rename(columns={year: "Mean NO2"})
        temp_df["Year"] = int(year)
        df = pd.concat([df, temp_df], ignore_index=True)

    wards = df["Council ward"].unique()
    colors = plt.cm.get_cmap('tab20', len(wards))

    for i, ward in enumerate(wards):
        fig, ax = plt.subplots(figsize=(10, 6))
        ward_data = df[df["Council ward"] == ward]
        color = colors(i)  # Assign one color per ward

        ax.scatter(
            ward_data["Year"], ward_data["Mean NO2"], color=color, alpha=0.7, s=15
        )
        ax.hlines(40, 2003, 2024, color='red', linestyle='--', label='2005 WHO Annual Mean Guideline (40 µg/m³)')
        ax.hlines(10, 2003, 2024, color='blue', linestyle='--', label='2021 WHO Annual Mean Guideline (10 µg/m³)')

        for site in ward_data["Site name"].unique():
            site_data = ward_data[ward_data["Site name"] == site]
            ax.plot(site_data["Year"].astype(int), site_data["Mean NO2"], color=color, alpha=0.7)

        

        ax.set_ylim(0, 100)
        ax.set_xlabel("Year")
        ax.set_ylabel("Mean NO₂ (µg/m³)")
        ax.set_title(f"NO₂ Over Time - {ward}")
        ax.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.tight_layout()
        plt.show()

def load_air_quality_data():
    file_path = pl.Path('datasets') / 'air_quality_monitoring_results_2003_to_2024_1.xlsx'
    dataframe = pd.read_excel(file_path, header=1) 
    dataframe = dataframe.reset_index(drop=True)
    
    dataframe.dropna(subset=["Council ward"], inplace=True)
    for col in dataframe.columns:
        dataframe[col] = dataframe[col].replace(0.0, np.nan) if str(col).startswith("2") else dataframe[col]
    
    dataframe.to_csv('air_quality_data.csv', index=False)

    """
    Splits the dataframe into a dictionary of dataframes by 'Council ward'.
    Returns: dict of {ward_name: dataframe}
    """
    dataframe["Council ward"] = dataframe["Council ward"].str.strip()
    dataframe["Council ward"] = dataframe["Council ward"].str.replace("Nether Edge& Sharrow", "Nether Edge & Sharrow")
    
    grouped = dataframe.groupby('Council ward')

    # If there are less than 3 entries for a ward, group them into "Other"
    for ward, group in grouped:
        if len(group) < 3:  # Threshold for "Other"
            dataframe.loc[dataframe["Council ward"] == ward, "Council ward"] = "Other"

    return dataframe

def load_predicted_data():
    file_path_2021 = pl.Path('datasets') / 'sheffield_2021_prediction.csv'
    file_path_2024 = pl.Path('datasets') / 'sheffield_2024_prediction.csv'
    file_path_2040 = pl.Path('datasets') / 'sheffield_2040_prediction.csv'

    dataframe21 = pd.read_csv(file_path_2021, header=3)
    dataframe24 = pd.read_csv(file_path_2024, header=3)
    dataframe40 = pd.read_csv(file_path_2040, header=3)

    cols_to_drop = ['geo_area', 'Local_Auth_Code', 'EU_zone_agglom_01']
    for df in [dataframe21, dataframe24, dataframe40]:
        df.drop(columns=cols_to_drop, inplace=True)

    return dataframe21, dataframe24, dataframe40

def trend_over_time(merged_dataframe):
    # Find year columns (assuming they are integers from 2005 to 2024)
    year_cols = [col for col in range(2005, 2025)]
    xs = list(range(2005, 2025))
    ys_matrix = merged_dataframe[year_cols].astype(float).values
    for ys in ys_matrix:
        plt.plot(xs, ys)

    # Add WHO limits
    plt.hlines(y=40, xmin=2005, xmax=2024, colors='black', linestyles='dashed', label='40 µg/m³ Limit')
    plt.hlines(y=10, xmin=2005, xmax=2024, colors='black', linestyles='dotted', label='10 µg/m³ Limit')
    
    plt.xlabel('Year')
    plt.ylabel('Nitrogen Dioxide Levels')
    plt.title('Nitrogen Dioxide Levels Over Time by Location')
    plt.xticks(xs)  # Force integer ticks for each year
    plt.legend()
    plt.show()
 

if __name__ == "__main__":
    dataframe = load_air_quality_data()
    trend_over_time(dataframe)
   # plot_ward_averages(dataframe)
    #plot_sites_where_over_2005_limit(dataframe)
    #plot_no2_over_time_by_ward(dataframe)
    #plot_site_locations_colour_by_ward(dataframe)
    #plot_site_locations(dataframe)

    #df21, df24, df40 = load_predicted_data()
    #plot_1km_grid_coloured_by_no2(df21, "Total_NO2_21", 2021)
    #plot_1km_grid_coloured_by_no2(df24, "Total_NO2_24", 2024)
    #plot_1km_grid_coloured_by_no2(df40, "Total_NO2_40", 2040)