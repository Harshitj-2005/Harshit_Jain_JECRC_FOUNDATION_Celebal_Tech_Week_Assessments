import pandas as pd
import numpy as np
import os

def load_and_aggregate_data(data_dir='data'):
  
    daily_dataset_path = os.path.join(data_dir, 'daily_dataset.csv')
    weather_path = os.path.join(data_dir, 'weather_daily_darksky.csv')
    households_path = os.path.join(data_dir, 'informations_households.csv')
    
    print("Loading datasets...")
    df_daily = pd.read_csv(daily_dataset_path)
    df_weather = pd.read_csv(weather_path)
    df_households = pd.read_csv(households_path)
    
    
    df_daily['day'] = pd.to_datetime(df_daily['day'])
    df_daily = df_daily.dropna(subset=['energy_sum'])
    
   
    df_daily = df_daily[(df_daily['day'] >= '2011-12-02') & (df_daily['day'] <= '2014-02-27')]
    
   
    df_weather['time'] = pd.to_datetime(df_weather['time'].str.split().str[0])
    df_weather['temperature'] = (df_weather['temperatureMax'] + df_weather['temperatureMin']) / 2
    
    df_weather_clean = df_weather[['time', 'temperature', 'humidity']].drop_duplicates(subset=['time'])
    
   
    city_daily_agg = df_daily.groupby('day')['energy_sum'].mean().reset_index()
    city_daily_agg.rename(columns={'day': 'datetime', 'energy_sum': 'energy_kwh'}, inplace=True)
    
  
    daily_city = pd.merge(city_daily_agg, df_weather_clean, left_on='datetime', right_on='time', how='left')
    daily_city.drop(columns=['time'], inplace=True)
    
    daily_city['temperature'] = daily_city['temperature'].ffill().bfill()
    daily_city['humidity'] = daily_city['humidity'].ffill().bfill()
    
  
    df_merged = pd.merge(df_daily, df_households, on='LCLid', how='inner')
    
    
    df_merged = df_merged[~df_merged['Acorn'].isin(['ACORN-', 'ACORN-U']) & df_merged['Acorn'].notna()]
    
    
    acorn_daily_agg = df_merged.groupby(['Acorn', 'day'])['energy_sum'].mean().reset_index()
    acorn_daily_agg.rename(columns={'Acorn': 'acorn_group', 'day': 'datetime', 'energy_sum': 'energy_kwh'}, inplace=True)
    
    
    acorn_daily = pd.merge(acorn_daily_agg, df_weather_clean, left_on='datetime', right_on='time', how='left')
    acorn_daily.drop(columns=['time'], inplace=True)
    acorn_daily['temperature'] = acorn_daily['temperature'].ffill().bfill()
    acorn_daily['humidity'] = acorn_daily['humidity'].ffill().bfill()
    
    return daily_city, acorn_daily

if __name__ == "__main__":
    
    for f in ['data/hourly_city.csv', 'data/acorn_hourly.csv']:
        if os.path.exists(f):
            try:
                os.remove(f)
                print(f"Removed old synthetic file: {f}")
            except Exception as e:
                print(f"Could not remove {f}: {e}")
                
    daily_city, acorn_daily = load_and_aggregate_data()
    
    daily_city.to_csv('data/daily_city.csv', index=False)
    acorn_daily.to_csv('data/acorn_daily.csv', index=False)
    print("Processed datasets successfully saved:")
    print("  - data/daily_city.csv")
    print("  - data/acorn_daily.csv")