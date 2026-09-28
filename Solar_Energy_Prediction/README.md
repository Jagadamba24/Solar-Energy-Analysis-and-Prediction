# ☀️ BAD702 – Solar Energy Generation Prediction

An end-to-end predictive analytics and statistical machine learning project designed for **BAD702**. This system analyzes real-world photovoltaic telemetry and weather variables to clean raw sensor data, conduct inferential statistical tests, build regression and discriminant models, and deploy an interactive **Streamlit** web application (**SolarSense**).

---

## 📁 Project Directory Structure

```text
Solar_Energy_Prediction/
│
├── dataset/
│   ├── solar_data.csv          # Raw uncleaned sensor telemetry (8,792 records)
│   └── solar_data_clean.csv    # Sanitized dataset after interpolation & outlier removal
│
├── notebooks/
│   └── solar_analysis.ipynb    # Comprehensive Jupyter Notebook covering Steps 3 to 19
│
├── app.py                      # Production Streamlit Web Application ("SolarSense")
├── model.pkl                   # Serialized ML pipeline, regression & LDA models, and metrics
├── requirements.txt            # Project dependencies
└── README.md                   # Full academic report and usage documentation
```

---

## 📊 Dataset Overview

The dataset contains hourly solar generation and meteorological sensor observations over a 1-year period (representing a **50 kWp** commercial photovoltaic rooftop system):

| Variable | Type | Unit | Description |
| :--- | :--- | :--- | :--- |
| **Date** | Date String | YYYY-MM-DD | Observation calendar date |
| **Time** | Time String | HH:MM | Observation timestamp (hourly) |
| **Solar_Irradiance** | Float | W/m² | Global Horizontal Irradiance (GHI) |
| **Temperature** | Float | °C | Ambient dry-bulb temperature |
| **Humidity** | Float | % | Atmospheric relative humidity |
| **Wind_Speed** | Float | m/s | Wind velocity at array height (cools panels) |
| **Cloud_Cover** | Float | % | Sky obscuration fraction |
| **Solar_Energy** | Float | kWh | **Target:** Total generated electrical energy |

---

## 🔬 Methodology & Module Breakdown

### Module 1 — Data Cleaning & Exploratory Data Analysis (EDA)
1. **Raw Telemetry Ingestion (Step 3):** Loaded 8,792 raw records with missing values, duplicate logs, and sensor glitches.
2. **Data Cleaning (Step 4):**
   - **Missing Values:** Imputed via linear time-series interpolation (`.interpolate()`), preserving diurnal continuity.
   - **Duplicates:** Pruned 32 duplicate sensor records (`.drop_duplicates()`).
   - **Datatypes:** Unified `Date` and `Time` into ISO-8601 `Datetime`; cast environmental variables to `float64`.
   - **Outlier Filtering:** Applied domain physics boundaries ($0 \le \text{Irradiance} \le 1350$ W/m², $-10 \le \text{Temp} \le 55$ °C, $0 \le \text{Cloud} \le 100$ %, $\text{Energy} \ge 0$).
3. **Descriptive Statistics (Step 5):** Calculated Mean, Median, Mode, Variance, Standard Deviation, Minimum, Maximum, Q1, Q3, and IQR.
4. **EDA Visualizations (Step 6):**
   - Target generation histogram & density estimation
   - Boxplots and violin distributions of meteorological covariates
   - Diurnal 24-hour cycle generation curves
   - Scatter plot of Solar Irradiance vs. Solar Generation
   - Pearson correlation heatmap ($r = +0.994$ between Irradiance and Energy)

### Module 2 — Sampling & Distributions
1. **Random Sampling (Step 7):** Extracted random sample ($n = 100$) from population ($N = 8,733$); computed sample mean $\bar{x} = 9.58$ kWh and sample standard deviation $s = 12.43$ kWh.
2. **Sampling Distribution (Step 8):** Drew $k = 1,000$ repeated samples ($n = 100$); demonstrated the **Central Limit Theorem (CLT)** where empirical standard error matches theoretical $SE = \frac{\sigma}{\sqrt{n}} \approx 1.17$ kWh.
3. **Confidence Interval (Step 9):** Constructed 95% Confidence Interval for mean hourly generation:
   $$\text{CI}_{95\%} = [7.11\text{ kWh},\ 12.04\text{ kWh}] \quad (\text{Margin of Error } = \pm 2.47\text{ kWh})$$

### Module 3 — Hypothesis Testing
1. **Research Question & Hypotheses:** *Does solar energy generation significantly differ between high-irradiance and low-irradiance daytime periods?*
   - $H_0: \mu_{\text{high}} = \mu_{\text{low}}$ (No difference)
   - $H_1: \mu_{\text{high}} \neq \mu_{\text{low}}$ (Significant difference)
2. **Two-Sample Welch's t-Test:**
   - High Irradiance Mean: $27.16$ kWh ($n = 2,173$)
   - Low Irradiance Mean: $11.32$ kWh ($n = 2,173$)
   - $t = 101.47$, $p < 0.0001$ ($\alpha = 0.05$)
   - **Decision:** Decisively **Reject $H_0$**, confirming irradiance as the fundamental driver of power yield.

### Module 4 — Regression Modeling
1. **Simple Linear Regression (Step 12):**
   $$\hat{y} = 0.2675 + 0.0373 \times \text{Solar\_Irradiance} \quad (R^2 = 0.9941)$$
2. **Multiple Linear Regression (Step 13):**
   - Features: Irradiance ($+0.038$), Temperature ($-0.024$), Humidity ($+0.011$), Wind Speed ($+0.013$), Cloud Cover ($+0.002$)
   - Intercept: $-0.1377$
   - $R^2 = 0.9948$, $\text{Adjusted } R^2 = 0.9948$, $\text{MAE} = 0.6344$ kWh, $\text{RMSE} = 0.8419$ kWh
   - Correctly captured **negative thermal derating** and **positive wind convective cooling**.
3. **Residual Diagnostics (Step 14):** Confirmed zero-mean, homoscedastic residual errors across predicted generation levels.

### Module 5 — Discriminant Analysis (LDA)
1. **Discretization (Step 15):** Partitioned generation into operational tiers:
   - **Low Generation:** $\le 5.0$ kWh
   - **Medium Generation:** $5.0 < Y \le 20.0$ kWh
   - **High Generation:** $> 20.0$ kWh
2. **Linear Discriminant Analysis (Step 16 & 17):**
   - Overall Accuracy: **96.05%**
   - Weighted Precision: **0.9617**, Recall: **0.9605**, F1-Score: **0.9591**

### Final ML Part — Machine Learning Benchmarking
Evaluated 3 regressors under an 80/20 train/test split:

| Model | R² Score | MAE (kWh) | RMSE (kWh) | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Regression** | 0.9948 | 0.6344 | 0.8419 | Parametric Baseline |
| **Random Forest Regressor** | **0.9989** | **0.1855** | **0.3799** | 🏆 **Top Champion** |
| **Gradient Boosting Regressor** | 0.9990 | 0.1819 | 0.3723 | Competitive Alternative |

- **Feature Importance:** Solar Irradiance ($99.8\%$), Temperature ($0.14\%$), Humidity ($0.03\%$), Cloud Cover ($0.02\%$), Wind Speed ($0.01\%$).

---

## 🌐 Streamlit Web Application ("SolarSense")

The project includes an interactive web platform built with Streamlit and Plotly:
- **Sidebar Navigation:** 11 dedicated pages mapping 1-to-1 with the project syllabus:
  1. `🏠 Home` — Executive overview, pipeline flow, KPI cards
  2. `📁 Dataset` — Interactive raw data viewer and variable glossary
  3. `🧹 Data Cleaning` — Quality audit tabs (missing, duplicates, datatypes, outliers)
  4. `📊 EDA` — Interactive Plotly graphs (scatter, diurnal curve, heatmap, histograms)
  5. `📈 Statistical Analysis` — Descriptive stats, CLT interactive simulator, 95% CI
  6. `🔬 Hypothesis Testing` — Two-sample t-test results and operational interpretation
  7. `📉 Regression` — Simple OLS, Multiple Regression, and Residual diagnostics
  8. `🎯 Classification` — LDA confusion matrix and regime classification
  9. `🔮 ML Prediction` — **Live What-If Solar Calculator** with quick scenario presets
  10. `🏆 Model Performance` — Benchmark table, actual vs. predicted plot, feature importance
  11. `📝 Conclusion` — Formal academic synthesis answering all Step 23 & 24 questions

---

## 🚀 How to Run Locally

### 1. Activate the Virtual Environment
On Windows (PowerShell):
```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Launch the Streamlit Application
```powershell
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 3. Open the Jupyter Notebook
```powershell
jupyter notebook notebooks/solar_analysis.ipynb
```

---

## 📝 Academic Conclusion & Key Findings
1. **Strongest Predictor:** Solar Irradiance accounts for over 99.7% of predictive power.
2. **Thermal Derating:** Ambient temperature has a verified negative coefficient, reflecting photovoltaic semiconductor efficiency reduction at elevated temperatures.
3. **Statistical Significance:** Hypothesis testing rejected $H_0$ with $p < 0.0001$, proving statistically distinct generation regimes between high and low irradiance daylight hours.
4. **Machine Learning Champion:** The **Random Forest Regressor** achieved superior generalization ($R^2 = 0.9989$, $\text{MAE} = 0.1855$ kWh), accurately modeling non-linear environmental interactions.
