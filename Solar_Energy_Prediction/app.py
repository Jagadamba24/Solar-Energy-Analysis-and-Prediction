import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import os

# =========================================================================
# PAGE CONFIGURATION
# =========================================================================
st.set_page_config(
    page_title="SolarSense | Solar Energy Generation Prediction",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================================
# CUSTOM CSS FOR ULTRA-PREMIUM AESTHETICS
# =========================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Modern dark theme background */
    .stApp {
        background: radial-gradient(circle at top right, #1a2332 0%, #0d131f 50%, #080b12 100%);
        color: #f1f5f9;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0b111e;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    /* Glassmorphism Metric Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 16px;
        backdrop-filter: blur(12px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: rgba(245, 158, 11, 0.4);
    }
    .metric-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #94a3b8;
        margin-bottom: 6px;
    }
    .metric-val {
        font-size: 2.1rem;
        font-weight: 700;
        background: linear-gradient(135deg, #f59e0b, #fbbf24);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-sub {
        font-size: 0.82rem;
        color: #64748b;
        margin-top: 4px;
    }
    
    /* Section Headings */
    .section-header {
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: #f8fafc;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .section-sub {
        font-size: 0.98rem;
        color: #94a3b8;
        margin-bottom: 24px;
        line-height: 1.5;
    }
    
    /* Highlight Badges */
    .badge-amber {
        display: inline-block;
        background: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.3);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 12px;
    }
    
    .badge-category-high {
        background: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
        padding: 8px 18px;
        border-radius: 12px;
        font-size: 1.2rem;
        font-weight: 700;
        display: inline-block;
    }
    .badge-category-med {
        background: rgba(245, 158, 11, 0.2);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.4);
        padding: 8px 18px;
        border-radius: 12px;
        font-size: 1.2rem;
        font-weight: 700;
        display: inline-block;
    }
    .badge-category-low {
        background: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
        padding: 8px 18px;
        border-radius: 12px;
        font-size: 1.2rem;
        font-weight: 700;
        display: inline-block;
    }
    
    /* Buttons */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: #0b0f19;
        font-weight: 700;
        border-radius: 10px;
        border: none;
        padding: 12px 28px;
        font-size: 1rem;
        transition: all 0.2s ease;
        box-shadow: 0 4px 14px rgba(245, 158, 11, 0.35);
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(245, 158, 11, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# =========================================================================
# DATA & MODEL CACHING
def categorize_energy(val):
    if val <= 5.0:
        return "Low Generation"
    elif val <= 20.0:
        return "Medium Generation"
    else:
        return "High Generation"

@st.cache_data
def load_datasets():
    # Load raw and cleaned data
    raw_path = "dataset/solar_data.csv" if os.path.exists("dataset/solar_data.csv") else "Solar_Energy_Prediction/dataset/solar_data.csv"
    clean_path = "dataset/solar_data_clean.csv" if os.path.exists("dataset/solar_data_clean.csv") else "Solar_Energy_Prediction/dataset/solar_data_clean.csv"
    
    raw = pd.read_csv(raw_path)
    clean = pd.read_csv(clean_path)
    clean["Datetime"] = pd.to_datetime(clean["Datetime"])
    clean["Generation_Category"] = clean["Solar_Energy"].apply(categorize_energy)
    return raw, clean

@st.cache_resource
def load_model_bundle():
    model_path = "model.pkl" if os.path.exists("model.pkl") else "Solar_Energy_Prediction/model.pkl"
    if os.path.exists(model_path):
        try:
            with open(model_path, "rb") as f:
                bundle = pickle.load(f)
            return bundle
        except Exception:
            # Handle incompatible scikit-learn pickle dtypes (e.g. node array dtype mismatch across sklearn versions)
            pass
    
    # Dynamically train fresh bundle matching the current runtime environment
    try:
        from train_and_evaluate import train_model_bundle
    except ImportError:
        from Solar_Energy_Prediction.train_and_evaluate import train_model_bundle
        
    bundle = train_model_bundle(save_to_disk=True, verbose=False)
    return bundle

raw_df, clean_df = load_datasets()
bundle = load_model_bundle()

# =========================================================================
# SIDEBAR NAVIGATION
# =========================================================================
with st.sidebar:
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 18px;">
            <div style="font-size: 2.2rem;">☀️</div>
            <div>
                <div style="font-size: 1.4rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.5px;">SolarSense</div>
                <div style="font-size: 0.75rem; color: #f59e0b; font-weight: 600; text-transform: uppercase;">Solar Analytics & Forecasting</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    menu = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "📁 Dataset",
            "🧹 Data Cleaning",
            "📊 EDA",
            "📈 Statistical Analysis",
            "🔬 Hypothesis Testing",
            "📉 Regression",
            "🎯 Classification",
            "🔮 ML Prediction",
            "🏆 Model Performance",
            "📝 Conclusion"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.78rem; color: #64748b; line-height: 1.5;">
        <b>Solar Analytics Pipeline</b><br>
        • 8,792 Raw Sensor Records<br>
        • Hypothesis & OLS Regression<br>
        • LDA & Ensemble ML<br>
        • Capacity: 50 kWp Array
    </div>
    """, unsafe_allow_html=True)

# =========================================================================
# 1. HOME PAGE
# =========================================================================
if menu == "🏠 Home":
    st.markdown('<span class="badge-amber">INTELLIGENT SOLAR PLATFORM</span>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Solar Energy Generation Prediction System</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-sub">
        An end-to-end statistical and machine learning pipeline for real-world solar photovoltaic generation analysis.
        From raw meteorological telemetry cleaning to inferential hypothesis testing, discriminant classification, and non-linear regression forecasting.
    </div>
    """, unsafe_allow_html=True)
    
    # KPI Highlights
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Observations</div>
            <div class="metric-val">{len(raw_df):,}</div>
            <div class="metric-sub">8,792 raw & {len(clean_df):,} cleaned</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Mean Solar Output</div>
            <div class="metric-val">{clean_df['Solar_Energy'].mean():.2f} <span style="font-size: 1rem; color: #94a3b8;">kWh</span></div>
            <div class="metric-sub">Max: {clean_df['Solar_Energy'].max():.1f} kWh (50 kWp plant)</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Best Model R² Score</div>
            <div class="metric-val">{bundle['metrics']['rf']['r2']:.4f}</div>
            <div class="metric-sub">Random Forest Regressor</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">LDA Classification</div>
            <div class="metric-val">{bundle['metrics']['lda']['accuracy']*100:.1f}%</div>
            <div class="metric-sub">Low, Medium & High Regime</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    st.markdown("### 📈 Plant Generation Profile (24-Hour Average Diurnal Cycle)")
    
    clean_df_hourly = clean_df.copy()
    clean_df_hourly["Hour"] = clean_df_hourly["Datetime"].dt.hour
    hourly_summary = clean_df_hourly.groupby("Hour")[["Solar_Energy", "Solar_Irradiance"]].mean().reset_index()
    
    diurnal_fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    diurnal_fig.add_trace(
        go.Scatter(
            x=hourly_summary["Hour"],
            y=hourly_summary["Solar_Energy"],
            name="Solar Generation (kWh)",
            line=dict(color="#f59e0b", width=3),
            fill="tozeroy",
            fillcolor="rgba(245, 158, 11, 0.15)",
            mode="lines+markers",
            marker=dict(size=6)
        ),
        secondary_y=False
    )
    
    diurnal_fig.add_trace(
        go.Scatter(
            x=hourly_summary["Hour"],
            y=hourly_summary["Solar_Irradiance"],
            name="Solar Irradiance (W/m²)",
            line=dict(color="#38bdf8", width=2, dash="dash"),
            mode="lines"
        ),
        secondary_y=True
    )
    
    diurnal_fig.update_layout(
        template="plotly_dark",
        height=350,
        margin=dict(t=30, b=30, l=40, r=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(
            title="Hour of Day (00:00 - 23:00)",
            tickmode="linear",
            tick0=0,
            dtick=2,
            gridcolor="rgba(255, 255, 255, 0.05)"
        ),
        yaxis=dict(
            title=dict(text="Solar Generation (kWh)", font=dict(color="#f59e0b")),
            tickfont=dict(color="#f59e0b"),
            gridcolor="rgba(255, 255, 255, 0.05)"
        ),
        yaxis2=dict(
            title=dict(text="Solar Irradiance (W/m²)", font=dict(color="#38bdf8")),
            tickfont=dict(color="#38bdf8"),
            showgrid=False
        )
    )
    
    st.plotly_chart(diurnal_fig, use_container_width=True)
    
    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 16px;">
            <div style="font-size: 0.8rem; font-weight: 700; color: #f59e0b; text-transform: uppercase;">⚡ Solar PV Facility</div>
            <div style="font-size: 1.1rem; font-weight: 700; color: #f8fafc; margin: 4px 0;">50 kWp Installed Array</div>
            <div style="font-size: 0.82rem; color: #94a3b8; line-height: 1.4;">
                Monitored over a full 1-year annual cycle with 8,733 validated hourly telemetry readings.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_c2:
        st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 16px;">
            <div style="font-size: 0.8rem; font-weight: 700; color: #38bdf8; text-transform: uppercase;">📊 Statistical Verification</div>
            <div style="font-size: 1.1rem; font-weight: 700; color: #f8fafc; margin: 4px 0;">Welch's t-Test & CLT</div>
            <div style="font-size: 0.82rem; color: #94a3b8; line-height: 1.4;">
                Proven statistical significance (p < 0.0001) between high and low irradiance operating regimes.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_c3:
        st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 16px;">
            <div style="font-size: 0.8rem; font-weight: 700; color: #10b981; text-transform: uppercase;">🤖 Predictive Intelligence</div>
            <div style="font-size: 1.1rem; font-weight: 700; color: #f8fafc; margin: 4px 0;">Random Forest & LDA</div>
            <div style="font-size: 0.82rem; color: #94a3b8; line-height: 1.4;">
                High-precision output forecasting with 94.1% regression R² and 96.1% regime classification accuracy.
            </div>
        </div>
        """, unsafe_allow_html=True)

# =========================================================================
# 2. DATASET PAGE
# =========================================================================
elif menu == "📁 Dataset":
    st.markdown('<span class="badge-amber">DATA SOURCE & TELEMETRY</span>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Raw Solar Generation Dataset Explorer</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-sub">
        Inspection of raw solar telemetry prior to preprocessing. Contains real-world sensor imperfections, unparsed timestamp strings, and meteorological covariates.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rows", f"{raw_df.shape[0]:,}")
    col2.metric("Total Columns", raw_df.shape[1])
    col3.metric("Missing Values", raw_df.isnull().sum().sum())
    col4.metric("Duplicate Rows", raw_df.duplicated().sum())
    
    st.markdown("### 📋 First 10 Records (Raw Telemetry)")
    st.dataframe(raw_df.head(10), use_container_width=True)
    
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("### 🏷️ Variable Dictionary & Data Types")
        schema_df = pd.DataFrame({
            "Column": raw_df.columns,
            "Raw Data Type": [str(t) for t in raw_df.dtypes],
            "Physical Unit": ["Date (YYYY-MM-DD)", "Time (HH:MM)", "W/m²", "°C", "%", "m/s", "%", "kWh (Target)"],
            "Description": [
                "Observation calendar date",
                "Observation hour and minute",
                "Global Horizontal Irradiance (GHI)",
                "Ambient dry-bulb temperature",
                "Atmospheric relative humidity",
                "Wind velocity at 10m height",
                "Fraction of sky obscured by clouds",
                "Net electrical energy generated by 50 kWp solar PV array"
            ]
        })
        st.dataframe(schema_df, use_container_width=True)
        
    with col_r:
        st.markdown("### 🔍 Raw Dataset Statistical Snapshot")
        st.dataframe(raw_df.describe().T.round(2), use_container_width=True)

# =========================================================================
# 3. DATA CLEANING PAGE
# =========================================================================
elif menu == "🧹 Data Cleaning":
    st.markdown('<span class="badge-amber">QUALITY ASSURANCE PIPELINE</span>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Data Cleaning & Quality Assurance</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-sub">
        Comprehensive quality assurance pipeline to sanitize raw sensor inputs: missing value imputation, duplicate elimination, datatype casting, and physical boundary outlier detection.
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Missing Values", "Duplicate Rows", "Data Types", "Outlier Detection"
    ])
    
    with tab1:
        st.markdown("#### Handling Missing Sensor Values")
        st.markdown("Raw sensors experience transmission packet drops. We identify null counts and apply linear time-series interpolation.")
        null_counts = raw_df.isnull().sum()
        col1, col2 = st.columns([1, 2])
        with col1:
            null_table = pd.DataFrame({"Missing Count": null_counts, "Percentage (%)": (null_counts / len(raw_df) * 100).round(2)})
            st.dataframe(null_table, use_container_width=True)
        with col2:
            fig = px.bar(
                x=null_counts.index, y=null_counts.values,
                labels={"x": "Feature", "y": "Null Count"},
                title="Missing Value Count by Feature (Raw Data)",
                color_discrete_sequence=["#e74c3c"]
            )
            fig.update_layout(template="plotly_dark", height=320)
            st.plotly_chart(fig, use_container_width=True)
        st.success("✅ **Resolution:** Imputed via linear interpolation (`.interpolate(method='linear')`) followed by forward/backward fill to preserve diurnal continuity.")
        
    with tab2:
        st.markdown("#### Handling Duplicate Records")
        st.markdown("Telemetric retry protocols frequently produce redundant timestamp logs.")
        dups = raw_df.duplicated().sum()
        col1, col2 = st.columns(2)
        col1.metric("Duplicate Rows Detected", dups)
        col2.metric("Post-Deduplication Rows", len(raw_df) - dups)
        st.code("clean_df = raw_df.drop_duplicates().reset_index(drop=True)", language="python")
        st.success(f"✅ **Resolution:** Successfully identified and pruned {dups} duplicate records.")
        
    with tab3:
        st.markdown("#### Datatype Normalization & Indexing")
        st.markdown("Parsed split `Date` and `Time` strings into an ISO-8601 unified `Datetime` timestamp and standardized all environmental readings into IEEE 64-bit floats.")
        
        col_t1, col_t2 = st.columns([1.2, 1])
        with col_t1:
            st.markdown("##### 📋 Before vs. After Schema Transformation")
            type_compare_df = pd.DataFrame({
                "Variable": ["Date", "Time", "Solar_Irradiance", "Temperature", "Humidity", "Wind_Speed", "Cloud_Cover", "Solar_Energy"],
                "Raw Telemetry": ["object (String)", "object (String)", "mixed / object", "mixed / object", "mixed / object", "float64", "mixed / object", "mixed / object"],
                "Normalized Type": ["Unified Datetime", "Unified Datetime", "float64 (W/m²)", "float64 (°C)", "float64 (%)", "float64 (m/s)", "float64 (%)", "float64 (kWh)"],
                "Transformation": ["Parsed to ISO-8601", "Merged with Date", "Numeric Coercion", "Numeric Coercion", "Numeric Coercion", "Verified", "Numeric Coercion", "Target Coercion"]
            })
            st.dataframe(type_compare_df, use_container_width=True)
            
        with col_t2:
            st.markdown("##### 💻 Implementation Code")
            st.code("""# 1. Combine Date & Time into Datetime index
clean_df['Datetime'] = pd.to_datetime(
    clean_df['Date'] + ' ' + clean_df['Time']
)

# 2. Coerce environmental readings to float64
numeric_features = [
    'Solar_Irradiance', 'Temperature', 
    'Humidity', 'Wind_Speed', 
    'Cloud_Cover', 'Solar_Energy'
]
for col in numeric_features:
    clean_df[col] = pd.to_numeric(
        clean_df[col], errors='coerce'
    )""", language="python")
            st.success("✅ **Status:** All 6 numerical covariates coerced to float64; unified chronological Datetime index created.")
        
    with tab4:
        st.markdown("#### Outlier Detection & Boundary Filtering")
        st.markdown("Combined Interquartile Range (IQR) checks with physical solar engineering boundaries to sanitize sensor telemetry.")
        
        col_out1, col_out2 = st.columns([1.3, 1])
        
        with col_out1:
            st.markdown("##### 🛡️ Physical Boundary Rules")
            rules_df = pd.DataFrame({
                "Parameter": ["Solar Irradiance", "Temperature", "Cloud Cover", "Solar Energy Output"],
                "Permitted Range": ["0 to 1,350 W/m²", "-10°C to 55°C", "0% to 100%", "≥ 0.0 kWh"],
                "Glitch Detected": ["Negative drift (< 0) & spikes (> 1,361)", "Faulty -45°C & +88.5°C spikes", "Impossible > 100% logs", "Negative inverter readings"],
                "Physical Justification": [
                    "Cannot be negative; capped at top-of-atmosphere constant",
                    "Geographical regional ambient limits",
                    "Sky coverage fraction bounded between clear (0%) and overcast (100%)",
                    "PV inverters cannot generate negative electrical energy"
                ]
            })
            st.dataframe(rules_df, use_container_width=True)
            
            c_raw, c_rem, c_clean = st.columns(3)
            c_raw.metric("Raw Telemetry", f"{len(raw_df):,}")
            c_rem.metric("Anomalies Removed", f"{len(raw_df) - len(clean_df)}")
            c_clean.metric("Sanitized Records", f"{len(clean_df):,}")
            
        with col_out2:
            st.markdown("##### 📦 Anomaly Boxplot View")
            outlier_fig = px.box(
                raw_df, y=["Solar_Irradiance", "Temperature", "Cloud_Cover", "Solar_Energy"],
                title="Raw Telemetry Showing Sensor Spikes",
                color_discrete_sequence=["#f59e0b"]
            )
            outlier_fig.update_layout(template="plotly_dark", height=280, margin=dict(t=35, b=20, l=20, r=20))
            st.plotly_chart(outlier_fig, use_container_width=True)
            
        st.success(f"✅ **Cleaned Output:** **{len(clean_df):,} verified records** ready for statistical inference and predictive modeling.")

# =========================================================================
# 4. EDA PAGE
# =========================================================================
elif menu == "📊 EDA":
    st.markdown('<span class="badge-amber">EXPLORATORY DATA ANALYSIS</span>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Exploratory Data Analysis (EDA)</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-sub">
        Multivariate statistical distributions, diurnal profiles, seasonal variations, and cross-variable correlation structures.
    </div>
    """, unsafe_allow_html=True)
    
    eda_tab = st.selectbox(
        "Select Visualization",
        [
            "Actual Observation: Irradiance vs Solar Generation",
            "Correlation Heatmap",
            "Time vs Solar Generation (Diurnal Profile)",
            "Distribution of Solar Energy (Histogram & KDE)",
            "Boxplots of Weather Variables",
            "Monthly Solar Energy Generation (Bar Plot)",
            "Violin Plot of Solar Irradiance"
        ]
    )
    
    if "Irradiance vs Solar Generation" in eda_tab or "Scatter" in eda_tab:
        st.markdown("#### Solar Irradiance vs Solar Energy Generation")
        st.markdown("Direct examination of the physical relationship between irradiance ($W/m^2$) and electrical energy produced ($kWh$).")
        
        sample_scatter = clean_df.sample(n=min(2000, len(clean_df)), random_state=42)
        fig = px.scatter(
            sample_scatter, x="Solar_Irradiance", y="Solar_Energy",
            color="Temperature",
            color_continuous_scale="Viridis",
            labels={"Solar_Irradiance": "Solar Irradiance (W/m²)", "Solar_Energy": "Solar Energy Output (kWh)", "Temperature": "Temp (°C)"},
            title="Solar Irradiance vs. Generation Output (Color-coded by Temperature)",
            opacity=0.6
        )
        fig.update_layout(template="plotly_dark", height=500)
        st.plotly_chart(fig, use_container_width=True)
        st.info("💡 **Observation:** Near-perfect linear alignment ($r > 0.99$), with slight downward curvature at high temperatures reflecting solar cell thermal derating.")
        
    elif "Correlation Heatmap" in eda_tab:
        st.markdown("#### Pearson Correlation Matrix")
        corr = clean_df[["Solar_Irradiance", "Temperature", "Humidity", "Wind_Speed", "Cloud_Cover", "Solar_Energy"]].corr()
        
        fig = px.imshow(
            corr, text_auto=".3f", aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Correlation Matrix of Meteorological Covariates & Energy Generation"
        )
        fig.update_layout(template="plotly_dark", height=480)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        - **Irradiance & Solar Energy:** Strongest positive relationship ($r = +0.994$).
        - **Temperature & Irradiance:** Positively correlated ($r \approx +0.48$) due to diurnal solar heating.
        - **Humidity & Irradiance:** Inversely correlated ($r \approx -0.45$).
        """)
        
    elif "Diurnal Profile" in eda_tab:
        st.markdown("#### Diurnal 24-Hour Solar Generation Profile")
        clean_df["Hour"] = clean_df["Datetime"].dt.hour
        hourly = clean_df.groupby("Hour")["Solar_Energy"].agg(["mean", "std", "max"]).reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hourly["Hour"], y=hourly["mean"],
            mode="lines+markers",
            name="Mean Generation (kWh)",
            line=dict(color="#f59e0b", width=3)
        ))
        fig.add_trace(go.Scatter(
            x=hourly["Hour"], y=hourly["max"],
            mode="lines",
            name="Peak Generation (kWh)",
            line=dict(color="#38bdf8", dash="dash")
        ))
        fig.update_layout(
            title="Average Solar Generation Across 24 Hours of the Day",
            xaxis_title="Hour of Day (0:00 to 23:00)",
            yaxis_title="Solar Energy (kWh)",
            template="plotly_dark",
            height=480,
            xaxis=dict(tickmode="linear", tick0=0, dtick=1)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.info("💡 Generation initiates at ~06:00, peaks at solar noon (~12:00-13:00) at ~28 kWh mean (up to 44 kWh max), and terminates by 19:00.")
        
    elif "Histogram" in eda_tab:
        st.markdown("#### Distribution of Solar Energy Output")
        fig = px.histogram(
            clean_df, x="Solar_Energy", nbins=40, marginal="box",
            color_discrete_sequence=["#f59e0b"],
            title="Target Distribution: Solar Energy Generation (kWh)"
        )
        fig.update_layout(template="plotly_dark", height=480)
        st.plotly_chart(fig, use_container_width=True)
        st.info("💡 Right-skewed distribution with a substantial mass at 0.0 kWh (night hours 19:00 - 06:00).")
        
    elif "Boxplots" in eda_tab:
        st.markdown("#### Boxplots of Environmental Features")
        fig = px.box(
            clean_df, y=["Temperature", "Humidity", "Cloud_Cover"],
            title="Distribution and Quartiles of Weather Factors",
            color_discrete_sequence=["#38bdf8"]
        )
        fig.update_layout(template="plotly_dark", height=480)
        st.plotly_chart(fig, use_container_width=True)
        
    elif "Monthly" in eda_tab:
        st.markdown("#### Mean Solar Energy Generation by Month")
        clean_df["Month"] = clean_df["Datetime"].dt.month_name()
        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        monthly = clean_df.groupby("Month")["Solar_Energy"].mean().reindex(month_order).reset_index()
        
        fig = px.bar(
            monthly, x="Month", y="Solar_Energy",
            color="Solar_Energy",
            color_continuous_scale="solar",
            title="Monthly Average Solar Generation (kWh)"
        )
        fig.update_layout(template="plotly_dark", height=480)
        st.plotly_chart(fig, use_container_width=True)
        
    elif "Violin" in eda_tab:
        st.markdown("#### Violin Plot: Solar Irradiance Density")
        fig = px.violin(
            clean_df, y="Solar_Irradiance", box=True, points=False,
            color_discrete_sequence=["#fb923c"],
            title="Violin Plot of Global Horizontal Irradiance (GHI in W/m²)"
        )
        fig.update_layout(template="plotly_dark", height=480)
        st.plotly_chart(fig, use_container_width=True)

# =========================================================================
# 5. STATISTICAL ANALYSIS PAGE
# =========================================================================
elif menu == "📈 Statistical Analysis":
    st.markdown('<span class="badge-amber">DESCRIPTIVE & INFERENTIAL DISTRIBUTIONS</span>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Statistical Analysis & Distributions</div>', unsafe_allow_html=True)
    
    samp_info = bundle["sampling"]
    
    # KPI Scorecard
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Population Mean (μ)", f"{samp_info['pop_mean']:.2f} kWh")
    k2.metric("Population Std (σ)", f"{samp_info['pop_std']:.2f} kWh")
    k3.metric("95% Confidence Interval", f"[{samp_info['ci_lower']:.2f}, {samp_info['ci_upper']:.2f}]")
    k4.metric("Margin of Error (E)", f"± {samp_info['margin_of_error']:.2f} kWh")
    
    st.markdown("---")
    
    # 1. Descriptive Statistics Table
    with st.expander("📊 View Complete Descriptive Statistics Table", expanded=True):
        stats_df = pd.DataFrame(bundle["stats"]).T.round(2)
        st.dataframe(stats_df, use_container_width=True)
        
    st.markdown("---")
    
    # 2. Side-by-side CLT Simulator & Confidence Interval Visual Plot
    col_clt, col_ci = st.columns([1.1, 1])
    
    with col_clt:
        st.markdown("##### 🎲 Central Limit Theorem (CLT) Simulator")
        s_col1, s_col2 = st.columns(2)
        sample_n = s_col1.slider("Sample Size (n)", min_value=30, max_value=300, value=100, step=10)
        num_sims = s_col2.slider("Samples (k)", min_value=200, max_value=2000, value=1000, step=100)
        
        np.random.seed(42)
        pop_data = clean_df["Solar_Energy"].values
        sim_means = [np.mean(np.random.choice(pop_data, size=sample_n, replace=True)) for _ in range(num_sims)]
        
        emp_se = np.std(sim_means)
        theo_se = np.std(pop_data, ddof=1) / np.sqrt(sample_n)
        
        clt_fig = px.histogram(
            sim_means, nbins=30, histnorm="probability density",
            title=f"Sample Means Distribution (n={sample_n}, k={num_sims:,})",
            color_discrete_sequence=["#10b981"]
        )
        clt_fig.add_vline(x=np.mean(pop_data), line_dash="dash", line_color="#f59e0b", annotation_text=f"μ = {np.mean(pop_data):.2f}")
        clt_fig.update_layout(template="plotly_dark", height=280, margin=dict(t=35, b=20, l=20, r=20), xaxis_title="Sample Mean Energy (kWh)")
        st.plotly_chart(clt_fig, use_container_width=True)
        
        c1, c2 = st.columns(2)
        c1.metric("Theoretical SE (σ/√n)", f"{theo_se:.4f}")
        c2.metric("Empirical SE of Means", f"{emp_se:.4f}")
        
    with col_ci:
        st.markdown("##### 🎯 95% Confidence Interval Estimation")
        
        # Visual CI chart
        ci_fig = go.Figure()
        
        # Point estimate & CI interval line
        ci_fig.add_trace(go.Scatter(
            x=[samp_info['ci_lower'], samp_info['sample_mean'], samp_info['ci_upper']],
            y=[1, 1, 1],
            mode="lines+markers",
            marker=dict(size=[12, 16, 12], color=["#38bdf8", "#f59e0b", "#38bdf8"]),
            line=dict(color="#38bdf8", width=4),
            name="95% CI Interval"
        ))
        # Population mean marker
        ci_fig.add_vline(x=samp_info['pop_mean'], line_dash="dot", line_color="#10b981", annotation_text=f"Pop Mean: {samp_info['pop_mean']:.2f}")
        
        ci_fig.update_layout(
            title="95% Confidence Interval for Hourly Mean (kWh)",
            template="plotly_dark",
            height=280,
            yaxis=dict(showticklabels=False, showgrid=False),
            xaxis_title="Solar Generation (kWh)",
            margin=dict(t=35, b=20, l=20, r=20),
            showlegend=False
        )
        st.plotly_chart(ci_fig, use_container_width=True)
        
        ci_col1, ci_col2, ci_col3 = st.columns(3)
        ci_col1.metric("Lower Limit", f"{samp_info['ci_lower']:.2f} kWh")
        ci_col2.metric("Sample Mean", f"{samp_info['sample_mean']:.2f} kWh")
        ci_col3.metric("Upper Limit", f"{samp_info['ci_upper']:.2f} kWh")
        
    st.info("💡 **Takeaway:** Even though hourly solar generation is skewed by night-time zeros, the distribution of sample means is Gaussian (Central Limit Theorem), yielding a precise 95% Confidence Interval of **[7.11, 12.04] kWh**.")

# =========================================================================
# 6. HYPOTHESIS TESTING PAGE
# =========================================================================
elif menu == "🔬 Hypothesis Testing":
    st.markdown('<span class="badge-amber">INFERENTIAL HYPOTHESIS TESTING</span>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Two-Sample t-Test Analysis</div>', unsafe_allow_html=True)
    
    ht = bundle["hypothesis_test"]
    
    # KPI Scorecard
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("High Irradiance Mean", f"{ht['high_mean']:.2f} kWh")
    k2.metric("Low Irradiance Mean", f"{ht['low_mean']:.2f} kWh")
    k3.metric("t-Statistic", f"{ht['t_stat']:.2f}")
    k4.metric("p-Value", "< 0.0001", delta="Reject H₀ (α=0.05)")
    
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.markdown("##### 🎯 Research Question & Hypotheses")
        st.markdown("**Question:** *Does solar generation significantly differ between high and low irradiance daytime hours?*")
        
        c_null, c_alt = st.columns(2)
        with c_null:
            st.error("""
            **H₀ (Null Hypothesis)**  
            ### μ_high = μ_low  
            **No difference:** Mean solar energy generation is identical between high and low irradiance periods.
            """)
        with c_alt:
            st.success("""
            **H₁ (Alternative Hypothesis)**  
            ### μ_high ≠ μ_low  
            **Significant difference:** High irradiance generates significantly higher solar energy.
            """)
            
        st.info("ℹ️ **Statistical Test:** Welch's Independent Two-Sample t-Test • **Significance Level:** α = 0.05 (95% Confidence)")
        
        st.markdown("##### 📊 Test Results")
        res_df = pd.DataFrame({
            "Metric": ["High Irradiance Group", "Low Irradiance Group", "Mean Difference", "Degrees of Freedom", "Decision (α = 0.05)"],
            "Value": [
                f"{ht['high_mean']:.2f} kWh (n=2,173)",
                f"{ht['low_mean']:.2f} kWh (n=2,173)",
                f"{ht['high_mean'] - ht['low_mean']:.2f} kWh",
                "4,344 (Welch's df)",
                "REJECT H0 (p < 0.0001)"
            ]
        })
        st.dataframe(res_df, use_container_width=True)
        
    with col2:
        st.markdown("##### 📈 Group Comparison Visualization")
        day_df = clean_df[clean_df["Solar_Irradiance"] > 10].copy()
        med_irr = day_df["Solar_Irradiance"].median()
        day_df["Group"] = np.where(day_df["Solar_Irradiance"] > med_irr, "High Irradiance (> 520 W/m²)", "Low Irradiance (<= 520 W/m²)")
        
        box_ht = px.box(
            day_df, x="Group", y="Solar_Energy", color="Group",
            color_discrete_map={"High Irradiance (> 520 W/m²)": "#f59e0b", "Low Irradiance (<= 520 W/m²)": "#38bdf8"},
            labels={"Solar_Energy": "Solar Generation (kWh)", "Group": "Regime"},
            title="Solar Output Distribution by Irradiance Regime"
        )
        box_ht.update_layout(template="plotly_dark", height=320, showlegend=False, margin=dict(t=35, b=20, l=20, r=20))
        st.plotly_chart(box_ht, use_container_width=True)
        
    st.info(f"💡 **Takeaway:** Decisively reject H0 (t = {ht['t_stat']:.2f}, p < 0.0001). Solar generation under high irradiance averages **{ht['high_mean']:.2f} kWh** compared to **{ht['low_mean']:.2f} kWh** under lower daylight irradiance, validating irradiance as the primary driver.")

# =========================================================================
# 7. REGRESSION PAGE
# =========================================================================
elif menu == "📉 Regression":
    st.markdown('<span class="badge-amber">PARAMETRIC REGRESSION MODELING</span>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Linear Regression & Residual Analysis</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-sub">
        Parametric modeling of solar generation using Ordinary Least Squares (OLS): Simple Linear Regression, Multiple Linear Regression, and Residual Diagnostics.
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Simple Linear Regression", "Multiple Linear Regression", "Residual Analysis"])
    
    with tab1:
        st.markdown("#### Simple Linear Regression ($X$ = Irradiance, $Y$ = Generation)")
        slr_meta = bundle["metrics"]["slr"]
        
        st.markdown("**Fitted Ordinary Least Squares (OLS) Equation:**")
        b0_val = f"{slr_meta['intercept']:.4f}"
        b1_val = f"{slr_meta['slope']:.4f}"
        st.latex(r"\hat{y} = " + b0_val + " + (" + b1_val + r" \times \text{Solar\_Irradiance})")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Slope (b₁)", f"{slr_meta['slope']:.4f}")
        col2.metric("Intercept (b₀)", f"{slr_meta['intercept']:.4f}")
        col3.metric("R² Score", f"{slr_meta['r2']:.4f}")
        
        # Plot Simple Regression
        sample_reg = clean_df.sample(n=min(1500, len(clean_df)), random_state=42).sort_values("Solar_Irradiance")
        fig_slr = px.scatter(
            sample_reg, x="Solar_Irradiance", y="Solar_Energy",
            title=f"Simple Linear Regression Fit (R² = {slr_meta['r2']:.4f})",
            labels={"Solar_Irradiance": "Solar Irradiance (W/m²)", "Solar_Energy": "Solar Energy (kWh)"},
            opacity=0.3, color_discrete_sequence=["#38bdf8"]
        )
        x_vals = np.linspace(0, 1100, 100)
        y_vals = slr_meta["intercept"] + slr_meta["slope"] * x_vals
        fig_slr.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="lines", name="OLS Regression Line", line=dict(color="#ef4444", width=3)))
        fig_slr.update_layout(template="plotly_dark", height=420)
        st.plotly_chart(fig_slr, use_container_width=True)
        
    with tab2:
        st.markdown("#### Multiple Linear Regression (5 Weather Covariates)")
        mlr_meta = bundle["metrics"]["mlr"]
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("R² Score", f"{mlr_meta['r2']:.4f}")
        c2.metric("Adjusted R²", f"{mlr_meta['adj_r2']:.4f}")
        c3.metric("MAE", f"{mlr_meta['mae']:.4f} kWh")
        c4.metric("RMSE", f"{mlr_meta['rmse']:.4f} kWh")
        
        st.markdown("##### Regression Coefficients Table")
        coef_df = pd.DataFrame({
            "Feature": list(mlr_meta["coefs"].keys()),
            "Coefficient": list(mlr_meta["coefs"].values()),
            "Physical Interpretation": [
                "Dominant positive driver: +0.038 kWh per W/m² increase",
                "Thermal derating: -0.024 kWh per °C rise (efficiency loss)",
                "Atmospheric absorption & light scatter",
                "Convective surface cooling boosts solar panel efficiency",
                "Residual attenuation factor"
            ]
        })
        st.dataframe(coef_df, use_container_width=True)
        
        st.markdown("##### Multiple Linear Regression Visualizations")
        
        # Sample data for responsive plotting
        sample_mlr = clean_df.sample(n=min(1500, len(clean_df)), random_state=42).copy()
        X_mlr_sample = sample_mlr[bundle["feature_cols"]]
        sample_mlr["Predicted_Energy"] = bundle["mlr_model"].predict(X_mlr_sample)
        sample_mlr["Residual"] = sample_mlr["Solar_Energy"] - sample_mlr["Predicted_Energy"]
        
        col_mlr1, col_mlr2 = st.columns(2)
        
        with col_mlr1:
            # Graph 1: Actual vs. Predicted Generation with 45° Ideal Line
            fig_avp = px.scatter(
                sample_mlr, x="Solar_Energy", y="Predicted_Energy",
                color="Residual",
                color_continuous_scale="Turbo",
                title=f"Actual vs. Predicted Solar Energy (R² = {mlr_meta['r2']:.4f})",
                labels={
                    "Solar_Energy": "Actual Solar Energy (kWh)",
                    "Predicted_Energy": "Predicted Solar Energy (kWh)",
                    "Residual": "Residual (kWh)"
                },
                opacity=0.6,
                hover_data=["Solar_Irradiance", "Temperature"]
            )
            max_limit = max(sample_mlr["Solar_Energy"].max(), sample_mlr["Predicted_Energy"].max())
            fig_avp.add_trace(go.Scatter(
                x=[0, max_limit], y=[0, max_limit],
                mode="lines", name="Ideal Fit (y = x)",
                line=dict(color="#ef4444", width=2.5, dash="dash")
            ))
            fig_avp.update_layout(
                template="plotly_dark",
                height=420,
                legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02)
            )
            st.plotly_chart(fig_avp, use_container_width=True)
            st.caption("📌 **Actual vs. Predicted Plot:** Evaluates the 5-dimensional regression model. Points tightly clustered along the red dashed 45° line confirm high predictive precision ($R^2 = 0.9948$).")
            
        with col_mlr2:
            # Graph 2: Feature Coefficients Bar Chart (Direction & Magnitude)
            coef_plot_df = coef_df.copy()
            coef_plot_df["Impact"] = coef_plot_df["Coefficient"].apply(
                lambda x: "Positive Driver (+)" if x >= 0 else "Negative Derating (-)"
            )
            fig_coef = px.bar(
                coef_plot_df, x="Coefficient", y="Feature", orientation="h",
                color="Impact",
                color_discrete_map={
                    "Positive Driver (+)": "#10b981",
                    "Negative Derating (-)": "#ef4444"
                },
                title="Regression Coefficients (Effect on Generation)",
                labels={"Coefficient": "OLS Partial Slope Coefficient", "Feature": "Covariate"}
            )
            fig_coef.add_vline(x=0, line_color="#94a3b8", line_width=1)
            fig_coef.update_layout(template="plotly_dark", height=420)
            st.plotly_chart(fig_coef, use_container_width=True)
            st.caption("📌 **Coefficient Impact:** Illustrates individual feature sensitivity: Solar Irradiance dominates positively, while Temperature derates panel efficiency (-0.024 kWh/°C).")
            
        # Graph 3: 3D Multiple Regression Plane (Interactive Expander)
        with st.expander("🌐 3D Interactive Multiple Regression Plane (Irradiance × Temperature → Generation)", expanded=False):
            st.markdown("Visualizing the fitted regression plane over the two strongest physical covariates (holding Humidity, Wind Speed, and Cloud Cover constant at their means).")
            
            irr_vals = np.linspace(clean_df["Solar_Irradiance"].min(), clean_df["Solar_Irradiance"].max(), 30)
            temp_vals = np.linspace(clean_df["Temperature"].min(), clean_df["Temperature"].max(), 30)
            irr_grid, temp_grid = np.meshgrid(irr_vals, temp_vals)
            
            plane_features = pd.DataFrame({
                "Solar_Irradiance": irr_grid.ravel(),
                "Temperature": temp_grid.ravel(),
                "Humidity": clean_df["Humidity"].mean(),
                "Wind_Speed": clean_df["Wind_Speed"].mean(),
                "Cloud_Cover": clean_df["Cloud_Cover"].mean()
            })
            z_plane = bundle["mlr_model"].predict(plane_features).reshape(irr_grid.shape)
            
            fig_3d = go.Figure()
            # 3D points
            fig_3d.add_trace(go.Scatter3d(
                x=sample_mlr["Solar_Irradiance"],
                y=sample_mlr["Temperature"],
                z=sample_mlr["Solar_Energy"],
                mode="markers",
                marker=dict(size=2.5, color=sample_mlr["Solar_Energy"], colorscale="Viridis", opacity=0.45),
                name="Observed Data Points"
            ))
            # 3D regression plane
            fig_3d.add_trace(go.Surface(
                x=irr_vals, y=temp_vals, z=z_plane,
                colorscale="YlOrRd", opacity=0.65, showscale=False,
                name="Fitted OLS Plane"
            ))
            fig_3d.update_layout(
                title="3D Multiple Regression Hyperplane (Irradiance vs. Temperature vs. Generation)",
                scene=dict(
                    xaxis_title="Solar Irradiance (W/m²)",
                    yaxis_title="Temperature (°C)",
                    zaxis_title="Solar Energy (kWh)",
                    camera=dict(eye=dict(x=1.6, y=-1.6, z=1.2))
                ),
                template="plotly_dark",
                height=540,
                margin=dict(l=0, r=0, b=0, t=40)
            )
            st.plotly_chart(fig_3d, use_container_width=True)
            
    with tab3:
        st.markdown("#### Residual Diagnostics & Assumption Verification")
        st.markdown("Evaluating OLS assumptions: linearity, homoscedasticity (constant error variance), and zero-mean residual distribution.")
        
        X_feats = clean_df[["Solar_Irradiance", "Temperature", "Humidity", "Wind_Speed", "Cloud_Cover"]]
        y_true = clean_df["Solar_Energy"]
        y_pred = bundle["mlr_model"].predict(X_feats)
        residuals = y_true - y_pred
        
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            fig_res = px.scatter(
                x=y_pred, y=residuals,
                labels={"x": "Predicted Solar Energy (kWh)", "y": "Residual Error (Actual - Predicted)"},
                title="Residuals vs. Predicted Values",
                opacity=0.3, color_discrete_sequence=["#a855f7"]
            )
            fig_res.add_hline(y=0, line_dash="dash", line_color="red")
            fig_res.update_layout(template="plotly_dark", height=380)
            st.plotly_chart(fig_res, use_container_width=True)
            
        with col_res2:
            fig_hist_res = px.histogram(
                residuals, nbins=40,
                title="Histogram of Residual Errors",
                color_discrete_sequence=["#10b981"]
            )
            fig_hist_res.add_vline(x=0, line_dash="dash", line_color="red")
            fig_hist_res.update_layout(template="plotly_dark", height=380, xaxis_title="Residual Error")
            st.plotly_chart(fig_hist_res, use_container_width=True)
            
        st.info("💡 **Diagnostics Summary:** Residual errors are tightly clustered around zero with symmetric distribution. The slight boundary cluster at 0 kWh reflects night-time non-negative power generation truncation.")

# =========================================================================
# 8. CLASSIFICATION & LDA PAGE
# =========================================================================
elif menu == "🎯 Classification":
    st.markdown('<span class="badge-amber">SUPERVISED DISCRIMINANT ANALYSIS</span>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Linear Discriminant Analysis (LDA)</div>', unsafe_allow_html=True)
    
    lda_meta = bundle["metrics"]["lda"]
    
    # KPI Scorecard
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall Accuracy", f"{lda_meta['accuracy']*100:.1f}%")
    c2.metric("Weighted Precision", f"{lda_meta['precision']:.4f}")
    c3.metric("Weighted Recall", f"{lda_meta['recall']:.4f}")
    c4.metric("F1-Score", f"{lda_meta['f1']:.4f}")
    
    st.markdown("---")
    
    col_dist, col_cm = st.columns([1, 1.2])
    
    with col_dist:
        st.markdown("##### 🏷️ Generation Regimes Distribution")
        if "Generation_Category" not in clean_df.columns:
            clean_df["Generation_Category"] = clean_df["Solar_Energy"].apply(categorize_energy)
        regime_counts = clean_df["Generation_Category"].value_counts().reindex(["Low Generation", "Medium Generation", "High Generation"]).fillna(0)
        
        dist_fig = px.bar(
            x=["Low (≤5 kWh)", "Medium (5-20 kWh)", "High (>20 kWh)"],
            y=regime_counts.values,
            labels={"x": "Operational Regime", "y": "Observation Count"},
            color=["Low (≤5 kWh)", "Medium (5-20 kWh)", "High (>20 kWh)"],
            color_discrete_map={
                "Low (≤5 kWh)": "#ef4444",
                "Medium (5-20 kWh)": "#f59e0b",
                "High (>20 kWh)": "#10b981"
            },
            title="Solar Output by Regime Tiers"
        )
        dist_fig.update_layout(template="plotly_dark", height=320, showlegend=False, margin=dict(t=35, b=20, l=20, r=20))
        st.plotly_chart(dist_fig, use_container_width=True)
        
    with col_cm:
        st.markdown("##### 🔲 Confusion Matrix")
        cm = np.array(lda_meta["cm"])
        labels = ["Low", "Medium", "High"]
        
        cm_fig = px.imshow(
            cm, text_auto=True,
            x=labels, y=labels,
            labels=dict(x="Predicted Regime", y="Actual Regime", color="Count"),
            title=f"LDA Confusion Matrix (Accuracy: {lda_meta['accuracy']*100:.1f}%)",
            color_continuous_scale="Blues"
        )
        cm_fig.update_layout(template="plotly_dark", height=320, margin=dict(t=35, b=20, l=20, r=20))
        st.plotly_chart(cm_fig, use_container_width=True)
        
    st.info("💡 **Takeaway:** Linear Discriminant Analysis achieves **96.05% classification accuracy** across Low (≤5 kWh), Medium (5-20 kWh), and High (>20 kWh) generation regimes, with near-zero false classifications between distant classes.")

# =========================================================================
# 9. ML PREDICTION PAGE (INTERACTIVE CALCULATOR)
# =========================================================================
elif menu == "🔮 ML Prediction":
    st.markdown('<span class="badge-amber">INTERACTIVE SOLAR INFERENCE ENGINE</span>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Live Solar Generation Predictor</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-sub">
        Enter environmental and meteorological conditions to compute real-time solar photovoltaic generation output and operational regime classification.
    </div>
    """, unsafe_allow_html=True)
    
    # Preset scenarios
    st.markdown("##### ⚡ Quick Scenarios")
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    
    irr_default = 650.0
    temp_default = 28.0
    hum_default = 55.0
    wind_default = 4.2
    cld_default = 15.0
    
    if p_col1.button("☀️ Peak Summer Noon"):
        irr_default = 980.0
        temp_default = 38.5
        hum_default = 35.0
        wind_default = 3.5
        cld_default = 5.0
        
    if p_col2.button("⛅ Partly Cloudy Autumn"):
        irr_default = 450.0
        temp_default = 24.0
        hum_default = 65.0
        wind_default = 4.8
        cld_default = 45.0
        
    if p_col3.button("🌧️ Overcast Monsoon"):
        irr_default = 120.0
        temp_default = 22.0
        hum_default = 92.0
        wind_default = 6.5
        cld_default = 90.0
        
    if p_col4.button("🌙 Night Time"):
        irr_default = 0.0
        temp_default = 16.0
        hum_default = 85.0
        wind_default = 2.0
        cld_default = 20.0
        
    st.markdown("---")
    
    col_input, col_pred = st.columns([1.1, 1])
    
    with col_input:
        st.markdown("### 🎛️ Input Meteorological Variables")
        
        in_irr = st.slider("Solar Irradiance (W/m²)", 0.0, 1200.0, float(irr_default), 5.0, help="Global Horizontal Irradiance measured at panel plane")
        in_temp = st.slider("Ambient Temperature (°C)", -5.0, 50.0, float(temp_default), 0.5, help="Ambient dry-bulb temperature")
        in_hum = st.slider("Relative Humidity (%)", 10.0, 100.0, float(hum_default), 1.0, help="Atmospheric relative humidity")
        in_wind = st.slider("Wind Speed (m/s)", 0.0, 18.0, float(wind_default), 0.2, help="Wind speed at array height (cools panels)")
        in_cld = st.slider("Cloud Cover (%)", 0.0, 100.0, float(cld_default), 1.0, help="Sky obscuration percentage")
        
        predict_btn = st.button("☀️ PREDICT SOLAR GENERATION", use_container_width=True)
        
    with col_pred:
        st.markdown("### ⚡ Prediction Results")
        
        # Prepare input DataFrame
        input_data = pd.DataFrame([{
            "Solar_Irradiance": in_irr,
            "Temperature": in_temp,
            "Humidity": in_hum,
            "Wind_Speed": in_wind,
            "Cloud_Cover": in_cld
        }])
        
        # Predict using Random Forest (best model)
        pred_energy = float(bundle["best_model"].predict(input_data)[0])
        pred_energy = max(0.0, pred_energy)
        
        # Classify regime using LDA
        pred_cat = bundle["lda_model"].predict(input_data)[0]
        
        # Capacity utilization for 50 kWp plant
        cuf = min(100.0, (pred_energy / 50.0) * 100)
        
        badge_class = "badge-category-high" if "High" in pred_cat else ("badge-category-med" if "Medium" in pred_cat else "badge-category-low")
        
        st.markdown(f"""
        <div class="metric-card" style="border-color: rgba(245, 158, 11, 0.4); text-align: center; padding: 30px;">
            <div class="metric-title" style="font-size: 1rem;">☀️ Predicted Solar Energy Output</div>
            <div class="metric-val" style="font-size: 3.5rem; margin: 10px 0;">{pred_energy:.2f} <span style="font-size: 1.4rem; color: #94a3b8;">kWh</span></div>
            <div style="margin: 15px 0;">
                <span class="{badge_class}">{pred_cat.upper()}</span>
            </div>
            <div style="font-size: 0.95rem; color: #94a3b8; margin-top: 15px;">
                Plant Capacity Utilization: <b>{cuf:.1f}%</b> (Rated 50 kWp System)
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Model Comparison for this specific prediction
        reg_pred = float(bundle["mlr_model"].predict(input_data)[0])
        
        st.markdown("##### Model Projections Comparison:")
        m_col1, m_col2 = st.columns(2)
        m_col1.metric("Linear Regression", f"{max(0.0, reg_pred):.2f} kWh")
        m_col2.metric("Random Forest (Best)", f"{pred_energy:.2f} kWh")

# =========================================================================
# 10. MODEL PERFORMANCE PAGE
# =========================================================================
elif menu == "🏆 Model Performance":
    st.markdown('<span class="badge-amber">MACHINE LEARNING BENCHMARK</span>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Linear Regression vs. Random Forest Benchmark</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-sub">
        Head-to-head performance comparison between the baseline parametric Linear Regression model and the non-linear Random Forest Regressor.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📊 Benchmark Comparison Table")
    comp_df = pd.DataFrame(bundle["comparison_table"])
    comp_df = comp_df[comp_df["Model"].isin(["Linear Regression", "Random Forest"])].reset_index(drop=True)
    st.dataframe(comp_df, use_container_width=True)
    
    col_feat, col_scatter = st.columns([1, 1.2])
    
    with col_feat:
        st.markdown("### 🌳 Feature Importance (Random Forest)")
        feat_imp_df = pd.DataFrame({
            "Feature": list(bundle["feature_importance"].keys()),
            "Importance": list(bundle["feature_importance"].values())
        }).sort_values("Importance", ascending=True)
        
        fig_imp = px.bar(
            feat_imp_df, x="Importance", y="Feature", orientation="h",
            title="Gini Feature Importance Ranking",
            color="Importance",
            color_continuous_scale="Viridis"
        )
        fig_imp.update_layout(template="plotly_dark", height=350, margin=dict(t=35, b=20, l=20, r=20))
        st.plotly_chart(fig_imp, use_container_width=True)
        
    with col_scatter:
        st.markdown("### 🎯 Actual vs. Predicted (Random Forest)")
        X_test = clean_df[["Solar_Irradiance", "Temperature", "Humidity", "Wind_Speed", "Cloud_Cover"]]
        y_test = clean_df["Solar_Energy"]
        y_test_pred = bundle["rf_model"].predict(X_test)
        
        sub_sample = pd.DataFrame({"Actual": y_test, "Predicted": y_test_pred}).sample(n=min(1200, len(y_test)), random_state=42)
        
        fig_act = px.scatter(
            sub_sample, x="Actual", y="Predicted",
            labels={"Actual": "Actual Generation (kWh)", "Predicted": "Predicted Generation (kWh)"},
            title="Random Forest: Actual vs. Predicted Solar Output",
            opacity=0.4, color_discrete_sequence=["#f59e0b"]
        )
        max_v = max(sub_sample["Actual"].max(), sub_sample["Predicted"].max())
        fig_act.add_trace(go.Scatter(x=[0, max_v], y=[0, max_v], mode="lines", name="Ideal 1:1 Parity", line=dict(color="red", dash="dash")))
        fig_act.update_layout(template="plotly_dark", height=350, margin=dict(t=35, b=20, l=20, r=20))
        st.plotly_chart(fig_act, use_container_width=True)
        
    st.info("💡 **Benchmark Winner:** **Random Forest Regressor** achieves the lowest prediction error with **MAE = 0.1855 kWh** and **R² = 0.9989**, outperforming Linear Regression (MAE = 0.6344 kWh, R² = 0.9948) by capturing non-linear thermal derating interactions.")

# =========================================================================
# 11. CONCLUSION & FINDINGS PAGE
# =========================================================================
elif menu == "📝 Conclusion":
    st.markdown('<span class="badge-amber">ANALYTICAL FINDINGS & SYNTHESIS</span>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Project Findings & Formal Conclusion</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="section-sub">
        Formal synthesis of data science methodologies, statistical tests, machine learning models, and real-world domain takeaways.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🔍 Key Empirical Findings")
    
    findings = [
        ("1. Strongest Relationship Variable", "<b>Solar Irradiance</b> has the overwhelmingly dominant relationship with solar generation ($r = +0.994$, Gini importance > 99.7%). Photovoltaic physics dictates that electron excitation is directly proportional to incoming photons."),
        ("2. Presence & Resolution of Missing Values", "Yes, missing values were present in raw telemetry across Irradiance (35), Temperature (30), Humidity (25), Cloud Cover (30), and Energy (25). They were successfully sanitized using time-series linear interpolation and boundary verification."),
        ("3. Statistical Significance in Hypothesis Testing", "Yes, the independent two-sample t-test between high and low irradiance daylight periods yielded $t = 101.47$ with $p < 0.0001$. The null hypothesis ($H_0$) was decisively rejected, confirming substantial differences in operational yield."),
        ("4. Linear Regression Performance", "Simple Linear Regression yielded $R^2 = 0.9941$, while Multiple Linear Regression reached $R^2 = 0.9948$ with $\\text{MAE} = 0.6344$ kWh and $\\text{RMSE} = 0.8419$ kWh. It confirmed negative thermal derating ($-0.024$ kWh/°C) and wind cooling (+0.013 kWh/(m/s))."),
        ("5. Model with Lowest Prediction Error", "<b>Random Forest Regressor</b> produced the lowest prediction error with $\\text{MAE} = 0.1855$ kWh and $R^2 = 0.9989$, significantly outperforming Linear Regression ($\\text{MAE} = 0.6344$ kWh, $R^2 = 0.9948$)."),
        ("6. Variables Contributing Most to Prediction", "Solar Irradiance (99.8%) followed by Ambient Temperature (0.14%), Relative Humidity (0.03%), Cloud Cover (0.02%), and Wind Speed (0.01%).")
    ]
    
    for title, text in findings:
        st.markdown(f"""
        <div class="metric-card" style="margin-bottom: 12px; padding: 18px 24px;">
            <div style="font-weight: 700; color: #f59e0b; margin-bottom: 4px; font-size: 1.05rem;">{title}</div>
            <div style="color: #cbd5e1; font-size: 0.95rem; line-height: 1.6;">{text}</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    st.markdown("### 🎓 System Synthesis & Conclusion")
    st.markdown("""
    The project analyzed solar-energy generation using statistical and machine-learning techniques. Data preprocessing and EDA were performed to understand the characteristics and relationships within the dataset. Sampling, confidence intervals, and hypothesis testing were used for statistical analysis. Regression techniques were used to model solar-energy generation, while classification and discriminant analysis were used to categorize generation levels. Machine-learning models were then evaluated using appropriate performance metrics. The resulting system can be used as a data-driven tool for estimating solar-energy generation.
    """)
    
    st.markdown("""
    <div style="text-align: center; color: #64748b; font-size: 0.85rem; margin-top: 30px;">
        SolarSense Analytics Platform • Real-Time Generation Intelligence
    </div>
    """, unsafe_allow_html=True)
