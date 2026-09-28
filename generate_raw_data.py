import numpy as np
import pandas as pd
import os

# Set seed for reproducibility
np.random.seed(42)

# Generate hourly timestamps for 1 full year (365 days * 24 hours = 8760 records)
date_range = pd.date_range(start="2023-01-01 00:00", end="2023-12-31 23:00", freq="H")
n = len(date_range)

dates = date_range.strftime("%Y-%m-%d")
times = date_range.strftime("%H:%M")
day_of_year = date_range.dayofyear.values
hour = date_range.hour.values

# 1. Base solar irradiance calculation based on solar geometry
# Solar declination angle delta
delta = 23.45 * np.sin(np.deg2rad(360 / 365 * (day_of_year - 81)))
# Latitude (e.g. 28.6 degrees N - typical high solar region like New Delhi / Arizona / Rajasthan)
lat = 28.0
# Hour angle (solar noon at 12:00)
hour_angle = 15.0 * (hour - 12.0)
# Solar elevation angle (sin_alpha)
sin_alpha = np.sin(np.deg2rad(lat)) * np.sin(np.deg2rad(delta)) + \
            np.cos(np.deg2rad(lat)) * np.cos(np.deg2rad(delta)) * np.cos(np.deg2rad(hour_angle))

# Clear-sky irradiance (W/m^2)
clear_sky_ghi = np.maximum(0.0, 1050.0 * sin_alpha)
# Set night hours (when elevation <= 0) to exactly 0
clear_sky_ghi[sin_alpha <= 0] = 0.0

# 2. Cloud Cover (%) with temporal persistence and seasonal trends (monsoon/winter clouds)
# Base cloudiness with seasonal peaks around July-August
seasonal_cloud = 25.0 + 30.0 * np.sin(np.deg2rad(360 / 365 * (day_of_year - 150))) ** 2
# Hourly fluctuation
cloud_noise = np.random.normal(0, 15, size=n)
cloud_cover = np.clip(seasonal_cloud + cloud_noise, 0.0, 100.0)

# 3. Actual Global Horizontal Irradiance (GHI in W/m^2) attenuated by cloud cover
# Cloud extinction coefficient
cloud_factor = 1.0 - 0.75 * (cloud_cover / 100.0) ** 2.2
irradiance = clear_sky_ghi * cloud_factor + np.random.normal(0, 8.0, size=n)
# Force night hours to 0
irradiance[clear_sky_ghi == 0] = 0.0
irradiance = np.clip(irradiance, 0.0, 1150.0)

# 4. Temperature (deg C): diurnal cycle + seasonal variation
# Seasonal temp: coldest in Jan (~14C), warmest in June (~38C)
seasonal_temp = 26.0 - 12.0 * np.cos(np.deg2rad(360 / 365 * (day_of_year - 20)))
# Diurnal temp cycle: min at 5 AM, max at 14 PM
diurnal_temp = 6.0 * np.sin(np.deg2rad(15 * (hour - 8)))
temp_noise = np.random.normal(0, 1.8, size=n)
temperature = np.round(seasonal_temp + diurnal_temp + temp_noise, 1)

# 5. Relative Humidity (%): inversely related to temperature
base_humidity = 85.0 - (temperature - 10.0) * 1.5 + (cloud_cover * 0.25)
humidity_noise = np.random.normal(0, 4.0, size=n)
humidity = np.clip(np.round(base_humidity + humidity_noise, 1), 10.0, 98.0)

# 6. Wind Speed (m/s): Weibull-like distribution (average 3.5 m/s)
wind_speed = np.clip(np.round(np.random.weibull(2.0, size=n) * 3.8 + 0.5, 2), 0.2, 18.0)

# 7. Solar Power Generation (kWh) for a 50 kWp PV Rooftop Array:
# Power = Capacity * (G / 1000) * [1 - gamma * (T_cell - 25)] * system_efficiency
# Cell temp = Ambient temp + G * 0.03 - 0.5 * Wind_Speed
cell_temp = temperature + irradiance * 0.032 - (wind_speed - 2.0) * 0.4
temp_loss_factor = np.maximum(0.70, 1.0 - 0.004 * (cell_temp - 25.0))
# System derating factor (inverter, wiring, soiling) ~ 0.86
rated_capacity_kw = 50.0
power_kw = rated_capacity_kw * (irradiance / 1000.0) * temp_loss_factor * 0.86
# Generation for 1 hour = power in kW * 1h = kWh
solar_energy = np.maximum(0.0, power_kw + np.random.normal(0, 0.4, size=n))
solar_energy[irradiance == 0] = 0.0
solar_energy = np.round(solar_energy, 2)
irradiance = np.round(irradiance, 2)
cloud_cover = np.round(cloud_cover, 1)

# Assemble initial clean DataFrame
df = pd.DataFrame({
    "Date": dates,
    "Time": times,
    "Solar_Irradiance": irradiance,
    "Temperature": temperature,
    "Humidity": humidity,
    "Wind_Speed": wind_speed,
    "Cloud_Cover": cloud_cover,
    "Solar_Energy": solar_energy
})

# =========================================================================
# INTRODUCE REALISTIC RAW DATA DEFECTS (For Step 4 Data Cleaning Requirements)
# 1. Missing values (NaNs in ~150 random cells)
# 2. Duplicate rows (~35 duplicate sensor transmissions)
# 3. Trailing spaces / string types in a few numeric cells
# 4. Outliers (sensor noise glitches: negative irradiance, extreme spikes)
# =========================================================================

# 1. Missing values:
null_indices_irr = np.random.choice(n, size=35, replace=False)
df.loc[null_indices_irr, "Solar_Irradiance"] = np.nan

null_indices_temp = np.random.choice(n, size=30, replace=False)
df.loc[null_indices_temp, "Temperature"] = np.nan

null_indices_hum = np.random.choice(n, size=25, replace=False)
df.loc[null_indices_hum, "Humidity"] = np.nan

null_indices_cld = np.random.choice(n, size=30, replace=False)
df.loc[null_indices_cld, "Cloud_Cover"] = np.nan

null_indices_nrg = np.random.choice(n, size=25, replace=False)
df.loc[null_indices_nrg, "Solar_Energy"] = np.nan

# 2. Sensor Outliers:
# Negative irradiance sensor drift (physically invalid)
outlier_neg_irr = np.random.choice(n, size=6, replace=False)
df.loc[outlier_neg_irr, "Solar_Irradiance"] = -25.5

# Huge sensor spike in irradiance (2350 W/m2 - solar constant at top of atmosphere is only 1361 W/m2)
outlier_pos_irr = np.random.choice(n, size=5, replace=False)
df.loc[outlier_pos_irr, "Solar_Irradiance"] = 2450.0

# Sensor temperature spike (88.5 deg C - impossible ambient temperature)
outlier_temp = np.random.choice(n, size=4, replace=False)
df.loc[outlier_temp, "Temperature"] = 88.5

# Extreme negative temperature error (-45 deg C)
outlier_temp_neg = np.random.choice(n, size=3, replace=False)
df.loc[outlier_temp_neg, "Temperature"] = -45.0

# Impossible cloud cover > 100%
outlier_cld = np.random.choice(n, size=4, replace=False)
df.loc[outlier_cld, "Cloud_Cover"] = 185.0

# Impossible negative energy reading
outlier_nrg = np.random.choice(n, size=5, replace=False)
df.loc[outlier_nrg, "Solar_Energy"] = -12.0

# 3. Duplicate rows (simulate telemetry replay / retry duplicate logs)
dup_indices = np.random.choice(n, size=32, replace=False)
duplicate_rows = df.iloc[dup_indices].copy()
df = pd.concat([df, duplicate_rows], ignore_index=True)

# 4. Save to dataset/solar_data.csv and Solar_Energy_Prediction/dataset/solar_data.csv
os.makedirs("dataset", exist_ok=True)
os.makedirs("Solar_Energy_Prediction/dataset", exist_ok=True)

df.to_csv("dataset/solar_data.csv", index=False)
df.to_csv("Solar_Energy_Prediction/dataset/solar_data.csv", index=False)

print(f"Raw Solar Dataset created successfully!")
print(f"Total Rows: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(f"Missing values:\n{df.isnull().sum()}")
print(f"Duplicate rows: {df.duplicated().sum()}")
