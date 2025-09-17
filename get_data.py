
import pandas as pd
import pathlib as pl
import numpy as np

def load_data():
    file_path = pl.Path('datasets') / "SO2Data" / "mapso22023.csv"
    dataframe = pd.read_csv(file_path, header = 5)
    return dataframe

def compile_data(start_year, end_year):
    # Compile data from multiple years into a single DataFrame
    all_data = []
    for year in range(start_year, end_year):
        
        # change file_path
        file_path = pl.Path("OZONEData") / f"mapdgt1{year}.csv"
        df = pd.read_csv(file_path, header=5)
        
        # replace 0 and 'MISSING' with NaN
        df[['dgt1'+str(year)]] = df[['dgt1'+str(year)]].replace(0, np.nan)
        df[['dgt1'+str(year)]] = df[['dgt1'+str(year)]].replace('MISSING', np.nan)
        df = df[['x', 'y', f'dgt1{year}']]
        all_data.append(df)
    
    # merge data into one dataframe
    merged_dataframe = all_data[0]
    for dataframe in all_data[1:]:
        merged_dataframe = pd.merge(merged_dataframe, dataframe, on=['x', 'y'], how = 'inner')
    
    # change save path as required
    merged_dataframe.to_csv(pl.Path("OZONEData") / "compiled_OZONE_data.csv", index=False)

if __name__ == "__main__":
    compile_data()