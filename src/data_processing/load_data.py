import pandas as pd

def load_soil_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def load_weather_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def load_yield_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df
