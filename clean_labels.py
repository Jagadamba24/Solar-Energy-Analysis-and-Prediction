import re

with open("app.py", "r", encoding="utf-8") as f:
    content = f.read()

# Replacements list (original -> clean professional label)
replacements = [
    # Badges
    ('<span class="badge-amber">ACADEMIC PROJECT • BAD702</span>', '<span class="badge-amber">INTELLIGENT SOLAR PLATFORM</span>'),
    ('<span class="badge-amber">STEP 1 & STEP 3</span>', '<span class="badge-amber">DATA SOURCE & TELEMETRY</span>'),
    ('<span class="badge-amber">STEP 4</span>', '<span class="badge-amber">QUALITY ASSURANCE PIPELINE</span>'),
    ('<span class="badge-amber">STEP 6</span>', '<span class="badge-amber">EXPLORATORY DATA ANALYSIS</span>'),
    ('<span class="badge-amber">MODULE 2 • STEPS 5, 7, 8, 9</span>', '<span class="badge-amber">DESCRIPTIVE & INFERENTIAL DISTRIBUTIONS</span>'),
    ('<span class="badge-amber">MODULE 3 • STEPS 10 & 11</span>', '<span class="badge-amber">INFERENTIAL HYPOTHESIS TESTING</span>'),
    ('<span class="badge-amber">MODULE 4 • STEPS 12, 13, 14</span>', '<span class="badge-amber">PARAMETRIC REGRESSION MODELING</span>'),
    ('<span class="badge-amber">MODULE 5 • STEPS 15, 16, 17</span>', '<span class="badge-amber">SUPERVISED DISCRIMINANT ANALYSIS</span>'),
    ('<span class="badge-amber">STEP 21 • INTERACTIVE INFERENCE</span>', '<span class="badge-amber">INTERACTIVE SOLAR INFERENCE ENGINE</span>'),
    ('<span class="badge-amber">FINAL ML PART • STEPS 18 & 19</span>', '<span class="badge-amber">MACHINE LEARNING BENCHMARK</span>'),
    ('<span class="badge-amber">STEPS 23 & 24</span>', '<span class="badge-amber">ANALYTICAL FINDINGS & SYNTHESIS</span>'),
    
    # Sidebar
    ('BAD702 Solar Analytics', 'Solar Analytics & Forecasting'),
    ('<b>BAD702 Project Pipeline</b>', '<b>Solar Analytics Pipeline</b>'),
    
    # Section titles
    ('### 📊 Step 5 — Descriptive Statistics', '### 📊 Descriptive Statistics'),
    ('### 🎲 Step 7 & 8 — Sampling & Central Limit Theorem', '### 🎲 Sampling & Central Limit Theorem'),
    ('### 🎯 Step 9 — 95% Confidence Interval', '### 🎯 95% Confidence Interval for Mean Generation'),
    ('### ❓ Step 10 — Research Question & Hypotheses', '### 🎯 Research Question & Hypotheses'),
    ('#### Step 12 — Simple Linear Regression', '#### Simple Linear Regression'),
    ('#### Step 13 — Multiple Linear Regression', '#### Multiple Linear Regression'),
    ('#### Step 14 — Residual Diagnostics & Assumption Verification', '#### Residual Diagnostics & Assumption Verification'),
    ('### 🏷️ Step 15 — Generation Regime Discretization', '### 🏷️ Generation Regime Discretization'),
    ('### 🔲 Step 17 — Confusion Matrix', '### 🔲 Confusion Matrix & Performance'),
    ('### 🔍 Step 23 — Summary of Key Findings', '### 🔍 Key Empirical Findings'),
    ('### 🎓 Step 24 — Formal Conclusion', '### 🎓 System Synthesis & Conclusion'),
    
    # Footers & texts
    ('BAD702 Predictive Analytics & SML • Project Final Delivery • SolarSense Platform', 'SolarSense Analytics Platform • Real-Time Generation Intelligence'),
    ('- **Module 1:** Load raw dataset', '- **Data Ingestion & Cleaning:** Load raw dataset'),
    ('- **Module 2 & 3:** Inferential statistics', '- **Statistical Inference:** Sampling distributions, Central Limit Theorem'),
    ('- **Module 4 & 5:** Simple and Multiple Linear Regression', '- **Modeling & Classification:** Parametric OLS regression, residual diagnostics, and LDA'),
    ('Rigorous 4-step quality assurance pipeline', 'Comprehensive quality assurance pipeline'),
    
    # Tab labels
    ('tab1, tab2, tab3, tab4 = st.tabs([\n        "1. Missing Values", "2. Duplicate Rows", "3. Data Types", "4. Outlier Detection"\n    ])',
     'tab1, tab2, tab3, tab4 = st.tabs([\n        "Missing Values", "Duplicate Rows", "Data Types", "Outlier Detection"\n    ])'),
    
    # Dropdown in EDA
    ('"1. Actual Observation: Irradiance vs Solar Generation (Scatter Plot)"', '"Actual Observation: Irradiance vs Solar Generation"'),
    ('"2. Correlation Heatmap"', '"Correlation Heatmap"'),
    ('"3. Time vs Solar Generation (Diurnal Profile)"', '"Time vs Solar Generation (Diurnal Profile)"'),
    ('"4. Distribution of Solar Energy (Histogram & KDE)"', '"Distribution of Solar Energy (Histogram & KDE)"'),
    ('"5. Boxplots of Weather Variables"', '"Boxplots of Weather Variables"'),
    ('"6. Monthly Solar Energy Generation (Bar Plot)"', '"Monthly Solar Energy Generation (Bar Plot)"'),
    ('"7. Violin Plot of Solar Irradiance"', '"Violin Plot of Solar Irradiance"')
]

count = 0
for old, new in replacements:
    if old in content:
        content = content.replace(old, new)
        count += 1
    else:
        print(f"Warning: pattern not found: {old[:40]}...")

# Clean any remaining "Step X" or "Module X" occurrences
content = re.sub(r'Step\s*\d+\s*[—–-]\s*', '', content)

with open("app.py", "w", encoding="utf-8") as f:
    f.write(content)

with open("Solar_Energy_Prediction/app.py", "w", encoding="utf-8") as f:
    f.write(content)

print(f"\nSuccessfully updated {count} patterns in app.py and Solar_Energy_Prediction/app.py!")
