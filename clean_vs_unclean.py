import matplotlib.pyplot as plt
import pandas as pd
import cartopy.crs as ccrs
from cartopy.io.img_tiles import OSM

"""Binned off - we only have one year of Clean Air Zone data (as it came in during 2023)."""


def get_airzoned_data():
    airzoned = pd.read_excel("Air quality clean air zoned.xlsx", sheet_name = "Clean air zone", header = 1)
    airzoned.replace(0, pd.NA, inplace=True)
    return airzoned

def get_non_airzoned_data():
    non_airzoned_data = pd.read_excel("Air quality clean air zoned.xlsx", sheet_name = "Non-Clean air zone", header = 1)
    non_airzoned_data.replace(0, pd.NA, inplace=True)
    return non_airzoned_data

def plot_coloured_by_airzone(airzone, not_airzone):
    """
    Plots 1km grid boxes around each (x, y) location, coloured by total NO2.
    Assumes dataframe has columns: 'x', 'y', 'total NO2'
    """
    imagery = OSM()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=imagery.crs)
    ax.add_image(imagery, 14)

    xs, ys = airzone['x'], airzone['y']

    # Create 1km grid boxes centered on each (x, y)
    for xi, yi in zip(xs, ys):
        # Each box is 1km x 1km (500m offset from center)
        ax.scatter(xi, yi, color='blue', s=10, transform=ccrs.OSGB(approx=False))
    
    xs, ys = not_airzone['x'], not_airzone['y']

    # Create 1km grid boxes centered on each (x, y)
    for xi, yi in zip(xs, ys):
        # Each box is 1km x 1km (500m offset from center)
        ax.scatter(xi, yi, color='red', s=10, transform=ccrs.OSGB(approx=False))

    ax.set_title(f'Sites Coloured by Airzone')
    plt.show()

def plot_airzone_comparison(airzone, not_airzone):
    # for each year, plot a boxplot comparing airzoned and non-airzoned data
    years = list(range(2005, 2024))
    for year in years:
        airzoned_year = airzone[year].dropna().astype(float)
        non_airzoned_year = not_airzone[year].dropna().astype(float)
        
        plt.figure()
        plt.boxplot([airzoned_year, non_airzoned_year], labels=['Air Zoned', 'Non-Air Zoned'])
        plt.title(f'NO2 Levels Comparison in {year}')
        plt.ylabel('Total NO2 Levels')
        plt.show()

def line_plot_airzone_comparison(airzone, not_airzone):
    years = list(range(2005, 2024))
    airzoned_means = [airzone[year].dropna().astype(float).mean() for year in years]
    non_airzoned_means = [not_airzone[year].dropna().astype(float).mean() for year in years]

    plt.figure()
    plt.plot(years, airzoned_means, label='Air Zoned', marker='o')
    plt.plot(years, non_airzoned_means, label='Non-Air Zoned', marker='o')
    plt.title('Average NO2 Levels Over Time')
    plt.xlabel('Year')
    plt.ylabel('Average Total NO2 Levels')
    plt.legend()
    plt.show()

def plot_yearly_change_in_no2(airzone, not_airzone):
    years = list(range(2005, 2024))
    airzoned_changes = [airzone[year].dropna().astype(float).mean() for year in years]
    non_airzoned_changes = [not_airzone[year].dropna().astype(float).mean() for year in years]

    airzoned_yearly_change = [j - i for i, j in zip(airzoned_changes[:-1], airzoned_changes[1:])]
    non_airzoned_yearly_change = [j - i for i, j in zip(non_airzoned_changes[:-1], non_airzoned_changes[1:])]

    plt.figure()
    plt.plot(years[1:], airzoned_yearly_change, label='Air Zoned', marker='o')
    plt.plot(years[1:], non_airzoned_yearly_change, label='Non-Air Zoned', marker='o')
    plt.axhline(0, color='grey', linestyle='--')
    plt.title('Yearly Change in Average NO2 Levels')
    plt.xlabel('Year')
    plt.ylabel('Yearly Change in Average Total NO2 Levels')
    plt.legend()
    plt.show()
    
    


if __name__ == "__main__":
    airzoned = get_airzoned_data()
    non_airzoned = get_non_airzoned_data()
    plot_yearly_change_in_no2(airzoned, non_airzoned)
    #plot_airzone_comparison(airzoned, non_airzoned)
    #plot_coloured_by_airzone(airzoned, non_airzoned)