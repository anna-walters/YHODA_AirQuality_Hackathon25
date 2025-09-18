"""Basic Functions to compile data from multiple years and plot trends over time."""

import pandas as pd
import pathlib as pl
import numpy as np
import matplotlib.pyplot as plt

def compile_data(pollutant: str, start_year: int, end_year:int, year_function):
    """Compile data from multiple years into a single DataFrame.
    
    Args:
        pollutant (str): The pollutant type (e.g., "OZONE", "MO", "SO2", "PM25", "PM10", "NO2").
        start_year (int): The starting year for data compilation.
        end_year (int): The ending year for data compilation.
        year_function (function): A function that takes a year as input and returns the corresponding column name in the CSV files.
    """
    all_data = []
    for year in range(start_year, end_year):
        
        # CHANGES BASED ON CSV NAMES + FORMAT
        file_path = pl.Path(f"{pollutant}Data") / f"mapdgt1{year}.csv"
        df = pd.read_csv(file_path, header = 5)
        
        # replace 0 and 'MISSING' with NaN
        df[[year_function(year)]] = df[[year_function(year)]].replace(0, np.nan)
        df[[year_function(year)]] = df[[year_function(year)]].replace('MISSING', np.nan)
        
        # keep only x, y and year column
        df = df[['x', 'y', year_function(year)]]
        all_data.append(df)
    
    # merge data into one dataframe
    merged_dataframe = all_data[0]
    for dataframe in all_data[1:]:
        merged_dataframe = pd.merge(merged_dataframe, dataframe, on=['x', 'y'], how = 'inner')
    
    # change save path as required
    merged_dataframe.to_csv(pl.Path(f"{pollutant}Data") / f"compiled_{pollutant}_data.csv", index=False)



def trend_over_time(pollutant:str, df: pd.DataFrame, year_function):
    """Create a line graph showing trends over time for each site in the dataframe."""
    
    pollutants = {
        "Ozone": {"start": 2013, "end": 2023, "limit": 60},
        "Carbon Monoxide": {"start": 2005, "end": 2010, "limit": 10},
        "Sulfur Dioxide": {"start": 2015, "end": 2023, "limit": 40},
        "PM2.5": {"start": 2011, "end": 2023, "limit": 5},
        "PM10": {"start": 2010, "end": 2023, "limit": 15},
        "Nitrogen Dioxide": {"start": 2003, "end": 2024, "limit": 10},
    }
    
    year_start = pollutants[pollutant]["start"]
    year_end = pollutants[pollutant]["end"]
    limit = pollutants[pollutant]["limit"]

    # Get the year column names
    year_cols = [year_function(year) for year in range(year_start, year_end+1)]

    xs = list(range(year_start, year_end+1))
    ys_matrix = df[year_cols].astype(float).values
    
    # plot the data
    for ys in ys_matrix:
        plt.plot(xs, ys)

    # Plot the limit for the pollutant
    plt.hlines(y=limit, xmin=year_start, xmax=year_end, colors='r', linestyles='dashed', label= f'{limit} µg/m³ Limit')
    
    if pollutant == "Nitrogen Dioxide":
        # also add 2005 WHO limit of 40
        plt.hlines(y=40, xmin=2005, xmax=2024, colors='black', linestyles='dashed', label='40 µg/m³ Limit')

    plt.xlabel('Year')
    plt.ylabel(f'{pollutant} Levels (µg/m³)')
    plt.title(f'{pollutant} Levels Over Time by Site')
    plt.legend()
    plt.show()

def ozone(year:int):
        return f"dgt1{year}"

def co(year:int):
    return f"com8hr{year}"

def so2(year:int):
    return f"so2{year}"

def pm25(year:int):
    return f"pm25{year}g"

def pm10(year:int):
    return f"pm10{year}g"

def no2(year:int):
    return str(year)


if __name__ == "__main__":
    dfs = {
        #"Ozone": pd.read_csv(pl.Path("COMPILED_DATA")/"compiled_OZONE_data.csv"),
        #"Carbon Monoxide": pd.read_csv(pl.Path("COMPILED_DATA")/"compiled_MO_data.csv"),
        #"Sulfur Dioxide": pd.read_csv(pl.Path("COMPILED_DATA")/"compiled_so2_data.csv"),
        #"PM2.5": pd.read_csv(pl.Path("COMPILED_DATA")/"compiled_PM25_data.csv"),
        #"PM10": pd.read_csv(pl.Path("COMPILED_DATA")/"compiled_PM10_data.csv"),
        "Nitrogen Dioxide": pd.read_csv(pl.Path("COMPILED_DATA")/"air_quality_data.csv"),
    }

    fncs = {
        "Ozone": ozone,
        "Carbon Monoxide": co,
        "Sulfur Dioxide": so2,
        "PM2.5": pm25,
        "PM10": pm10,
        "Nitrogen Dioxide": no2,
    }

    for pollutant, df in dfs.items():
        df.replace('MISSING', np.nan, inplace=True)
        df.replace(0, np.nan, inplace=True)
        trend_over_time(pollutant, df, fncs[pollutant])