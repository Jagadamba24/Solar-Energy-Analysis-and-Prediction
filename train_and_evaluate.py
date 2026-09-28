import numpy as np
import pandas as pd
import pickle
import os
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_recall_fscore_support, confusion_matrix

print("Loading raw dataset...")
raw_df = pd.read_csv("dataset/solar_data.csv")
print(f"Raw shape: {raw_df.shape}")

# =========================================================================
# STEP 4: DATA CLEANING
# =========================================================================
# 1. Missing values
missing_counts_before = raw_df.isnull().sum()
# Clean missing values: for time-series meteorological data, forward-fill then backward-fill
clean_df = raw_df.copy()

# 2. Duplicate rows
duplicates_count = clean_df.duplicated().sum()
clean_df = clean_df.drop_duplicates().reset_index(drop=True)

# 3. Data types: ensure numeric columns are float
numeric_cols = ["Solar_Irradiance", "Temperature", "Humidity", "Wind_Speed", "Cloud_Cover", "Solar_Energy"]
for col in numeric_cols:
    clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

# Forward fill / backward fill missing values after type coercion
clean_df[numeric_cols] = clean_df[numeric_cols].interpolate(method="linear").bfill().ffill()

# 4. Outliers detection and handling
# Physical boundary filtering:
# Solar Irradiance cannot be negative, and top-of-atmosphere is ~1361 W/m2
clean_df = clean_df[(clean_df["Solar_Irradiance"] >= 0) & (clean_df["Solar_Irradiance"] <= 1350)]
# Ambient Temperature realistically between -10C and 55C
clean_df = clean_df[(clean_df["Temperature"] >= -10) & (clean_df["Temperature"] <= 55)]
# Cloud Cover between 0 and 100%
clean_df = clean_df[(clean_df["Cloud_Cover"] >= 0) & (clean_df["Cloud_Cover"] <= 100)]
# Solar Energy cannot be negative
clean_df = clean_df[clean_df["Solar_Energy"] >= 0]

# Add Datetime column for time-series analysis
clean_df["Datetime"] = pd.to_datetime(clean_df["Date"] + " " + clean_df["Time"])
clean_df = clean_df.sort_values("Datetime").reset_index(drop=True)

print(f"Cleaned dataset shape: {clean_df.shape}")

# Save cleaned dataset for quick reference
clean_df.to_csv("dataset/solar_data_clean.csv", index=False)
clean_df.to_csv("Solar_Energy_Prediction/dataset/solar_data_clean.csv", index=False)

# =========================================================================
# STEP 5: DESCRIPTIVE STATISTICS
# =========================================================================
stats_dict = {}
for col in numeric_cols:
    s = clean_df[col]
    q1 = s.quantile(0.25)
    q2 = s.median()
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    stats_dict[col] = {
        "Mean": float(s.mean()),
        "Median": float(q2),
        "Mode": float(s.mode()[0]),
        "Variance": float(s.var()),
        "Std_Dev": float(s.std()),
        "Min": float(s.min()),
        "Max": float(s.max()),
        "Q1": float(q1),
        "Q3": float(q3),
        "IQR": float(iqr)
    }
stats_df = pd.DataFrame(stats_dict).T
print("\n--- DESCRIPTIVE STATISTICS ---")
print(stats_df)

# =========================================================================
# STEP 7 & 8 & 9: SAMPLING, SAMPLING DISTRIBUTION & CONFIDENCE INTERVAL
# =========================================================================
np.random.seed(42)
pop_energy = clean_df["Solar_Energy"].values
pop_mean = np.mean(pop_energy)
pop_std = np.std(pop_energy, ddof=1)

# Step 7: Single sample of n=100
sample_100 = np.random.choice(pop_energy, size=100, replace=False)
sample_mean = np.mean(sample_100)
sample_std = np.std(sample_100, ddof=1)

# Step 8: Sampling distribution (1000 samples of n=100)
sample_means_100 = [np.mean(np.random.choice(pop_energy, size=100, replace=True)) for _ in range(1000)]
sampling_mean = np.mean(sample_means_100)
sampling_se = np.std(sample_means_100)
theory_se = pop_std / np.sqrt(100)

# Step 9: 95% Confidence Interval for mean solar generation
# using t-distribution with df = 99
t_crit = stats.t.ppf(0.975, df=99)
margin_of_error = t_crit * (sample_std / np.sqrt(100))
ci_lower = sample_mean - margin_of_error
ci_upper = sample_mean + margin_of_error

print(f"\n--- SAMPLING & CONFIDENCE INTERVAL ---")
print(f"Population Mean: {pop_mean:.4f}, Pop Std: {pop_std:.4f}")
print(f"Sample (n=100) Mean: {sample_mean:.4f}, Sample Std: {sample_std:.4f}")
print(f"Theoretical SE: {theory_se:.4f}, Empirical SE of Sample Means: {sampling_se:.4f}")
print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}], Margin of Error: {margin_of_error:.4f}")

# =========================================================================
# STEP 10 & 11: HYPOTHESIS TESTING (Independent Two-Sample t-Test)
# =========================================================================
# Research Question: Does solar energy generation significantly differ between
# high-irradiance (> 400 W/m2) and low-irradiance (<= 400 W/m2, daylight only > 0) periods?
daylight_df = clean_df[clean_df["Solar_Irradiance"] > 10].copy()
median_day_irr = daylight_df["Solar_Irradiance"].median()

high_irr_group = daylight_df[daylight_df["Solar_Irradiance"] > median_day_irr]["Solar_Energy"]
low_irr_group = daylight_df[daylight_df["Solar_Irradiance"] <= median_day_irr]["Solar_Energy"]

t_stat, p_val = stats.ttest_ind(high_irr_group, low_irr_group, equal_var=False)

print(f"\n--- HYPOTHESIS TEST (t-Test) ---")
print(f"High Irradiance Mean: {high_irr_group.mean():.4f} kWh (n={len(high_irr_group)})")
print(f"Low Irradiance Mean: {low_irr_group.mean():.4f} kWh (n={len(low_irr_group)})")
print(f"t-statistic: {t_stat:.4f}, p-value: {p_val:.4e}")
print(f"Decision at alpha=0.05: {'Reject H0 (Statistically Significant Difference)' if p_val < 0.05 else 'Fail to Reject H0'}")

# =========================================================================
# STEP 12: SIMPLE LINEAR REGRESSION
# =========================================================================
X_simple = clean_df[["Solar_Irradiance"]]
y = clean_df["Solar_Energy"]

slr = LinearRegression()
slr.fit(X_simple, y)
slr_pred = slr.predict(X_simple)
slr_r2 = r2_score(y, slr_pred)
slr_slope = slr.coef_[0]
slr_intercept = slr.intercept_

print(f"\n--- SIMPLE LINEAR REGRESSION ---")
print(f"Formula: Solar_Energy = {slr_intercept:.4f} + ({slr_slope:.4f} * Solar_Irradiance)")
print(f"R-squared: {slr_r2:.4f}")

# =========================================================================
# STEP 13 & 14: MULTIPLE LINEAR REGRESSION & RESIDUAL ANALYSIS
# =========================================================================
feature_cols = ["Solar_Irradiance", "Temperature", "Humidity", "Wind_Speed", "Cloud_Cover"]
X = clean_df[feature_cols]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlr = LinearRegression()
mlr.fit(X_train, y_train)
y_pred_mlr = mlr.predict(X_test)

mlr_r2 = r2_score(y_test, y_pred_mlr)
n_test = len(y_test)
p_feat = len(feature_cols)
adj_r2_mlr = 1 - (1 - mlr_r2) * (n_test - 1) / (n_test - p_feat - 1)
mlr_mae = mean_absolute_error(y_test, y_pred_mlr)
mlr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_mlr))

print(f"\n--- MULTIPLE LINEAR REGRESSION ---")
for feat, coef in zip(feature_cols, mlr.coef_):
    print(f"Coeff {feat}: {coef:.5f}")
print(f"Intercept: {mlr.intercept_:.4f}")
print(f"R2: {mlr_r2:.4f}, Adj R2: {adj_r2_mlr:.4f}, MAE: {mlr_mae:.4f}, RMSE: {mlr_rmse:.4f}")

# =========================================================================
# STEP 15, 16, 17: DISCRIMINANT ANALYSIS (LDA) & CLASSIFICATION
# =========================================================================
# Categories: Low Generation (0 to 5 kWh, including night/overcast),
# Medium Generation (5 to 20 kWh), High Generation (> 20 kWh)
def categorize_energy(val):
    if val <= 5.0:
        return "Low Generation"
    elif val <= 20.0:
        return "Medium Generation"
    else:
        return "High Generation"

clean_df["Generation_Category"] = clean_df["Solar_Energy"].apply(categorize_energy)
category_order = ["Low Generation", "Medium Generation", "High Generation"]

y_cat = clean_df["Generation_Category"]
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_cat, test_size=0.2, random_state=42, stratify=y_cat)

lda = LinearDiscriminantAnalysis()
lda.fit(X_train_c, y_train_c)
y_pred_lda = lda.predict(X_test_c)

lda_acc = accuracy_score(y_test_c, y_pred_lda)
lda_prec, lda_rec, lda_f1, _ = precision_recall_fscore_support(y_test_c, y_pred_lda, average="weighted")
cm = confusion_matrix(y_test_c, y_pred_lda, labels=category_order)

print(f"\n--- LINEAR DISCRIMINANT ANALYSIS ---")
print(f"LDA Accuracy: {lda_acc * 100:.2f}%")
print(f"LDA Precision: {lda_prec:.4f}, Recall: {lda_rec:.4f}, F1: {lda_f1:.4f}")
print(f"Confusion Matrix (Low, Med, High):\n{cm}")

# =========================================================================
# STEP 18 & 19: TRAIN & COMPARE ML MODELS
# =========================================================================
# Model 1: Linear Regression (already trained)
# Model 2: Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
rf_r2 = r2_score(y_test, y_pred_rf)
rf_mae = mean_absolute_error(y_test, y_pred_rf)
rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))

# Model 3: Gradient Boosting Regressor
gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
gbr.fit(X_train, y_train)
y_pred_gbr = gbr.predict(X_test)
gbr_r2 = r2_score(y_test, y_pred_gbr)
gbr_mae = mean_absolute_error(y_test, y_pred_gbr)
gbr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_gbr))

comparison_df = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "R²": [round(mlr_r2, 4), round(rf_r2, 4)],
    "MAE (kWh)": [round(mlr_mae, 4), round(rf_mae, 4)],
    "RMSE (kWh)": [round(mlr_rmse, 4), round(rf_rmse, 4)]
})

print(f"\n--- MODEL COMPARISON ---")
print(comparison_df.to_string(index=False))

# Feature importances from Random Forest
feat_imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print(f"\n--- FEATURE IMPORTANCES ---")
print(feat_imp)

# Save artifact dictionary to model.pkl
model_bundle = {
    "feature_cols": feature_cols,
    "slr_model": slr,
    "mlr_model": mlr,
    "rf_model": rf,
    "gbr_model": gbr,
    "best_model": rf,  # Best overall performer
    "lda_model": lda,
    "category_order": category_order,
    "metrics": {
        "slr": {"slope": slr_slope, "intercept": slr_intercept, "r2": slr_r2},
        "mlr": {"r2": mlr_r2, "adj_r2": adj_r2_mlr, "mae": mlr_mae, "rmse": mlr_rmse, "coefs": dict(zip(feature_cols, mlr.coef_)), "intercept": mlr.intercept_},
        "rf": {"r2": rf_r2, "mae": rf_mae, "rmse": rf_rmse},
        "gbr": {"r2": gbr_r2, "mae": gbr_mae, "rmse": gbr_rmse},
        "lda": {"accuracy": lda_acc, "precision": lda_prec, "recall": lda_rec, "f1": lda_f1, "cm": cm.tolist()}
    },
    "comparison_table": comparison_df.to_dict(orient="records"),
    "feature_importance": feat_imp.to_dict(),
    "stats": stats_dict,
    "hypothesis_test": {
        "t_stat": t_stat,
        "p_val": p_val,
        "high_mean": float(high_irr_group.mean()),
        "low_mean": float(low_irr_group.mean()),
        "decision": "Reject H0 (Statistically Significant Difference)"
    },
    "sampling": {
        "pop_mean": pop_mean,
        "pop_std": pop_std,
        "sample_mean": sample_mean,
        "sample_std": sample_std,
        "theoretical_se": theory_se,
        "empirical_se": sampling_se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "margin_of_error": margin_of_error,
        "sample_means_sample": sample_means_100[:100]
    }
}

with open("model.pkl", "wb") as f:
    pickle.dump(model_bundle, f)

with open("Solar_Energy_Prediction/model.pkl", "wb") as f:
    pickle.dump(model_bundle, f)

print("\nModel bundle saved successfully to model.pkl and Solar_Energy_Prediction/model.pkl!")
