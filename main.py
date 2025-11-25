import pandas as pd
from src.data_processing.process_soil import process_soil
from src.model.train_model import train_xgboost_soil

df = pd.read_csv("data/raw/soil_raw.csv")
df_processed = process_soil(df)

model = train_xgboost_soil(df_processed)
