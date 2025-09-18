import matplotlib.pyplot as plt
import pandas as pd
import cartopy.crs as ccrs
from cartopy.io.img_tiles import OSM
import pathlib as pl


### IS THERE A SIGNIFICANT DIFFERENCE IN NO2 LEVELS BETWEEN SITES IN AIR ZONES AND NOT IN AIR ZONES?
## Not pursued - we only have one year of Clean Air Zone data (as it came into effect during 2023).

def get_airzoned_data():
    airzoned = pd.read_excel(pl.Path("COMPILED_DATA") / "Air quality clean air zoned.xlsx", sheet_name = "Clean air zone", header = 1)
    airzoned.replace(0, pd.NA, inplace=True)
    non_airzoned_data = pd.read_excel(pl.Path("COMPILED_DATA") / "Air quality clean air zoned.xlsx", sheet_name = "Non-Clean air zone", header = 1)
    non_airzoned_data.replace(0, pd.NA, inplace=True)
    return airzoned, non_airzoned_data

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


### HOW WOULD THE BIOWALLS IMPACT ON NO2 EFFECT THE PREDICTIONS + DATA?

def biowall_reduction(dataframe):
    no2_reduction = 0.715

    site_subset = dataframe[dataframe[2024] > 40]
    site_subset["no2_prediction"] = site_subset[2024] * (1 - no2_reduction)
    print(site_subset["no2_prediction"])

def plot_1km_grid_change(dataframe_21, dataframe_40):
    imagery = OSM()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=imagery.crs)
    ax.add_image(imagery, 14)

    combined_dataframe = pd.merge(dataframe_21, dataframe_40, on=['x', 'y'])
    
    # 70% reduction in one year from 2021 to 2022
    reduction = combined_dataframe['Total_NO2_21'] * 0.715 - combined_dataframe["Total_NO2_21"]
    combined_dataframe['change'] = combined_dataframe['Total_NO2_40'] - combined_dataframe['Total_NO2_21'] + reduction

    xs, ys, no2_40, change = combined_dataframe['x'], combined_dataframe['y'], combined_dataframe['Total_NO2_40'], combined_dataframe['change']

    for xi, yi, n40, ch in zip(xs, ys, no2_40, change):
        # Color logic: green if <40, red if >=40
        if n40 > 40:
            color = 'red'
        elif n40 <= 10:
            color = 'green'
        else:
            color = 'yellow'

        rect = plt.Rectangle(
            (xi - 500, yi - 500), 1000, 1000,
            linewidth=0.5,
            edgecolor='black',
            facecolor=color,
            alpha=0.7,
            transform=ccrs.OSGB(approx=False)
        )
        ax.add_patch(rect)
        # Add text for change, rounded to 1 decimal place
        ax.text(
            xi, yi, f"{ch:.1f}",
            color='black', fontsize=8, ha='center', va='center',
            transform=ccrs.OSGB(approx=False)
        )


    # Add legend/key for color coding
    import matplotlib.patches as mpatches
    legend_handles = [
        mpatches.Patch(color='green', label='2021 WHO Guideline (≤10 µg/m³)'),
        mpatches.Patch(color='yellow', label='2005 WHO Guideline (≤40 µg/m³)'),
        mpatches.Patch(color='red', label='Above 2005 Guideline (>40 µg/m³)')
    ]
    ax.legend(handles=legend_handles, loc='upper right', fontsize='small')

    ax.set_title(f'1km Grid Boxes Coloured by Total NO₂ - Change (2040 - 2021)')
    plt.show()
    
def plot_no2_reduction_using_percentage_change(dataframe):
    # Find the first year where there is data and reduce it by 71.5%

    dataframe2 = dataframe[dataframe[2024] > 40].copy()
    dataframe2 = dataframe2.reset_index(drop=True)

    # find the yearly percentage change for each site

    percentage_change = pd.DataFrame(columns = ["DEFRA site ID"] + [str(year) for year in range(2006, 2025)])
    percentage_change["DEFRA site ID"] = dataframe2["DEFRA site ID"]
    percentage_change = percentage_change.reset_index(drop=True)
    for _, row in dataframe2.iterrows():
        for year in range(2005, 2025):
            percentage_change.loc[percentage_change["DEFRA site ID"] == row["DEFRA site ID"], year] = (row[year] - row[year-1]) / row[year-1]


    # for first valid year, apply 71.5% reduction
    for _, row in dataframe2.iterrows():
        first_valid_year = None
        for year in range(2005, 2025):
            if not pd.isna(row[year]):
                first_valid_year = year
                break
        if first_valid_year:
            dataframe2.loc[dataframe2["DEFRA site ID"] == row["DEFRA site ID"], first_valid_year] = row[first_valid_year] * (1 - 0.715)
            # apply percentage change for subsequent years
            for year in range(first_valid_year + 1, 2025):
                if not pd.isna(row[year]):
                    dataframe2.loc[dataframe2["DEFRA site ID"] == row["DEFRA site ID"], year] = dataframe2.loc[dataframe2["DEFRA site ID"] == row["DEFRA site ID"], year - 1].values[0] * (1 + percentage_change.loc[percentage_change["DEFRA site ID"] == row["DEFRA site ID"], year].values[0])

    fig,ax = plt.subplots(figsize=(10, 6))
    for _, row in dataframe2.iterrows():
        ax.plot(range(2005, 2025), row[range(2005, 2025)], label=row["Site name"], color = 'blue', alpha=0.5)
    
    dataframe = dataframe[dataframe[2024] > 40]
    for _, row in dataframe.iterrows():
        ax.plot(range(2005, 2025), row[range(2005, 2025)], label=row["Site name"], color = 'red', alpha=0.3)

    ax.legend(["First Year reduced by 71.5%", "Original Data"], loc='upper right', fontsize='small')

    ax.hlines(40, 2005, 2024, color='red', linestyle='--', label='2005 WHO Annual Mean Guideline (40 µg/m³)')
    ax.hlines(10, 2005, 2024, color='blue', linestyle='--', label='2021 WHO Annual Mean Guideline (10 µg/m³)')
    ax.set_ylim(0, 100)
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean NO₂ (µg/m³)")
    ax.set_title("NO₂ Over Time - Sites Initially Over 40 µg/m³, Reduced by 71.5% in First Valid Year")
    plt.show()
    
    


if __name__ == "__main__":

    # Clean Air Zone
    airzoned, non_airzoned = get_airzoned_data()
    plot_yearly_change_in_no2(airzoned, non_airzoned)
    plot_airzone_comparison(airzoned, non_airzoned)
    plot_coloured_by_airzone(airzoned, non_airzoned)

    # Biowalls
    from NO2 import load_predicted_data, load_air_quality_data
    dataframe_21, dataframe_24, dataframe_40 = load_predicted_data()
    plot_1km_grid_change(dataframe_21, dataframe_40)
    
    dataframe_base = load_air_quality_data()
    biowall_reduction(dataframe_base)
    plot_no2_reduction_using_percentage_change(dataframe_base)



