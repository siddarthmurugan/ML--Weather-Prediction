# Implementation of Random Forest Algorithm for Weather Prediction
## AIM:
To write a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm.

## Problem Statement and Dataset



## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1) Load the weather dataset using pandas.
2) Preprocess the data by handling missing values and sorting by time.
3) Select features and create lag variables for temperature and PM2.5.
4) Train Random Forest models to predict temperature and PM2.5 and save the models.


## Program:

Program to implement the Random Forest Algorithm to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data.
Developed by: SIDDARTH.M.V
RegisterNumber:  212225230267

```
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load dataset
df = pd.read_csv("weather-station-eee-block_2024_07_13.csv")
df.columns = df.columns.str.strip()
df['time'] = pd.to_datetime(df['time'], errors='coerce')

print("Original rows:", len(df))

# Only drop if target missing
df = df.dropna(subset=['tem', 'pm2_5'])

# Fill feature columns instead of dropping
df['hum'] = df['hum'].fillna(df['hum'].mean())
df['pressure'] = df['pressure'].fillna(df['pressure'].mean())
df['wind_speed'] = df['wind_speed'].fillna(df['wind_speed'].mean())
df['co2'] = df['co2'].fillna(df['co2'].mean())

# Sort by time
df = df.sort_values('time')

# Create lag features
df['Temp_Lag1'] = df['tem'].shift(1)
df['PM_Lag1'] = df['pm2_5'].shift(1)

# Only remove first row created by shift
df = df.iloc[1:]

print("Rows after preprocessing:", len(df))

# Features
X = df[['hum', 'pressure', 'wind_speed', 'co2',
        'Temp_Lag1', 'PM_Lag1']]

y_temp = df['tem']
y_pm = df['pm2_5']

print("Training samples:", len(X))

# Train models
model_temp = RandomForestRegressor(n_estimators=300, random_state=42)
model_pm = RandomForestRegressor(n_estimators=300, random_state=42)

model_temp.fit(X, y_temp)
model_pm.fit(X, y_pm)

# Save models
joblib.dump(model_temp, "temperature_model.pkl")
joblib.dump(model_pm, "pm25_model.pkl")

print("Models trained and saved successfully!")

```



## Output:
<img width="1468" height="126" alt="Screenshot 2026-04-28 182459" src="https://github.com/user-attachments/assets/8e2c2be8-6a26-40ec-84a4-f1a7556376b7" />
<img width="1771" height="657" alt="Screenshot 2026-04-28 182529" src="https://github.com/user-attachments/assets/624dbc30-34b8-4201-acbe-4529592fa17f" />
<img width="1816" height="660" alt="Screenshot 2026-04-28 182556" src="https://github.com/user-attachments/assets/cfcd4caf-2341-490d-a3b6-3432bdd92950" />
<img width="1896" height="701" alt="Screenshot 2026-04-28 182621" src="https://github.com/user-attachments/assets/4f5084ea-dae0-4662-8895-97c3b64561d5" />
<img width="1427" height="265" alt="Screenshot 2026-04-28 182645" src="https://github.com/user-attachments/assets/aae81b62-1365-48e9-8516-391e11aebce9" />


## Result:
