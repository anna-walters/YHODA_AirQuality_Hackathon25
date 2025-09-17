
import pandas as pd
import matplotlib.pyplot as plt
import pathlib as pl
import numpy as np

def trend_over_time(merged_dataframe):
    # Find year columns (assuming they are integers from 2015 to 2024)
    year_cols = [f"dgt1{col}" for col in range(2013, 2024)]
    # plot a line graph with so2 over time for each (x, y) location
    xs = list(range(2013, 2024))
    ys_matrix = merged_dataframe[year_cols].astype(float).values
    for ys in ys_matrix:
        plt.plot(xs, ys)

    plt.hlines(y=60, xmin=2013, xmax=2023, colors='r', linestyles='dashed', label='60 µg/m³ Limit')
        
    plt.xlabel('Year')
    plt.ylabel('Ozone Levels')
    plt.title('Ozone Levels Over Time by Location')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv(pl.Path("COMPILED_DATA")/"compiled_OZONE_data.csv")
    df.replace('MISSING', np.nan, inplace=True)
    trend_over_time(df)