# Time-Series Prediction of Urban Air Pollutant Concentration: CO(GT)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Overview

This project aims to build and evaluate machine learning models for predicting the concentration of Carbon Monoxide (`CO(GT)`) based on the [Air Quality Data Set](https://archive.ics.uci.edu/dataset/360/air+quality) from the UCI Machine Learning Repository.

The analysis involves:
* Data loading (via `ucimlrepo`) and extensive cleaning/preprocessing.
* Exploratory Data Analysis (EDA) to understand trends, seasonality, and correlations.
* Comprehensive feature engineering (time-based, lags, rolling stats, interactions, Fourier transforms).
* Advanced missing data imputation using `IterativeImputer`.
* Training and evaluation of multiple models:
    * Baseline: Linear Regression, Random Forest
    * Advanced: XGBoost, LightGBM, LSTM
* Comparison of model performance and identification of key predictive features.

**Target Variable:** `CO(GT)`

---

## Table of Contents

* [Project Structure](#project-structure)
* [Setup and Installation](#setup-and-installation)
* [Usage](#usage)
* [Project Workflow](#project-workflow)
* [Results Summary](#results-summary)
* [Detailed Findings](#detailed-findings)
* [Future Work](#future-work)
* [License](#license)

---

## Project Structure

air-quality-time-series-prediction/
│
├── .gitignore                # Git ignore file
├── LICENSE                   # Project license (e.g., MIT)
├── README.md                 # This file (main project overview)
│
├── data/
│   └── processed/            # Stores cleaned data (created by notebooks)
│       └── 01_air_quality_cleaned.pkl
│
├── models/                   # Stores saved model artifacts and scalers (created by notebooks)
│   ├── scalers/
│   │   ├── standard_scaler_adv.joblib
│   │   ├── minmax_scaler_adv.joblib
│   │   └── y_scaler_lstm.joblib
│   ├── linear_regression_basic.joblib
│   ├── random_forest_initial.joblib
│   ├── random_forest_optimized.joblib # (If tuning was performed)
│   ├── xgboost_adv.joblib
│   ├── lightgbm_adv.joblib
│   └── lstm_adv.keras
│
├── notebooks/                # Jupyter notebooks for exploration & workflow demonstration
│   ├── 01_Data_Loading_Preprocessing.ipynb
│   ├── 02_Exploratory_Data_Analysis.ipynb
│   ├── 03_Feature_Eng_Basic_Models.ipynb
│   └── 04_Advanced_Features_Models.ipynb
│
├── reports/                  # Generated outputs, figures, and detailed summaries
│   ├── figures/              # Stores key plots (e.g., EDA, predictions)
│   │   ├── 02_target_timeseries.png
│   │   ├── 02_correlation_matrix.png
│   │   └── ... (other saved plots) ...
│   ├── 03_basic_model_results.csv
│   ├── 04_advanced_model_results.csv
│   └── project_summary.md    # Detailed markdown report of all phases & findings
│
├── requirements.txt          # Python package dependencies
│
└── src/                      # Source code (.py files) with reusable functions
├── init.py           # Makes 'src' a Python package
├── evaluate.py           # Functions for calculating metrics and plotting results
├── feature_engineering.py # Functions for creating features & imputation
├── model_training.py     # Functions for training models & splitting data
├── preprocessing.py      # Functions for loading, cleaning, datetime handling
└── utils.py              # Helper functions (plotting setup, saving/loading objects)

*(Note: The `data/processed/`, `models/`, and `reports/figures/` directories are typically created dynamically by the notebooks when saving outputs. Ensure your `.gitignore` excludes large model files if necessary.)*

---

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/air-quality-time-series-prediction.git](https://github.com/your-username/air-quality-time-series-prediction.git) # Replace with your repo URL
    cd air-quality-time-series-prediction
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

The primary workflow and analysis are detailed in the Jupyter notebooks located in the `notebooks/` directory. It's recommended to run them sequentially from the project's root directory:

1.  `notebooks/01_Data_Loading_Preprocessing.ipynb`: Fetches, cleans, and preprocesses the data. Saves cleaned data to `data/processed/`.
2.  `notebooks/02_Exploratory_Data_Analysis.ipynb`: Performs EDA on the cleaned data. Saves key plots to `reports/figures/`.
3.  `notebooks/03_Feature_Eng_Basic_Models.ipynb`: Implements initial feature engineering and trains/evaluates baseline models. Saves results and optionally models.
4.  `notebooks/04_Advanced_Features_Models.ipynb`: Implements advanced techniques and trains/evaluates advanced models. Saves results, plots, models, and scalers.

Reusable functions for each step are located in the `src/` directory and are imported by the notebooks.

---

## Project Workflow

1.  **Data Loading & Preprocessing (`src/preprocessing.py`, Notebook 01):** Fetched data via `ucimlrepo`, handled -200 placeholders, created `DateTime` index, converted types, performed initial `ffill`/`bfill` imputation.
2.  **Exploratory Data Analysis (Notebook 02):** Visualized time series, distributions, correlations, seasonality, diurnal/weekly patterns.
3.  **Feature Engineering (`src/feature_engineering.py`, Notebooks 03 & 04):** Created time-based, lagged, rolling, interaction, and Fourier features. Used `IterativeImputer` for advanced imputation.
4.  **Modeling (`src/model_training.py`, Notebooks 03 & 04):** Trained Linear Regression, Random Forest (with optional tuning), XGBoost, LightGBM, and LSTM models on chronologically split data. Applied feature scaling where appropriate.
5.  **Evaluation (`src/evaluate.py`, Notebooks 03 & 04):** Assessed models using MAE, RMSE, R2. Visualized predictions and residuals.

---

## Results Summary

* **Best Performing Model:** **XGBoost** demonstrated the best performance after advanced feature engineering and imputation.
    * **RMSE:** ~0.49
    * **R2:** ~0.87
* **Gradient Boosting:** LightGBM also performed well (RMSE: ~0.50, R2: ~0.87).
* **LSTM:** The implemented LSTM model performed poorly (Negative R2), requiring significant further investigation.
* **Key Features:** Lagged target values, specific sensor readings (e.g., `PT08 S1(CO)`), and time-based features (e.g., `Hour`, Fourier components) were generally most important for the tree-based models.

**Advanced Model Performance (from Notebook 04):**

| Model    |      MAE |     RMSE |         R2 |
|:---------|---------:|---------:|-----------:|
| XGBoost  | 0.345317 | 0.489226 |   0.873602 |
| LightGBM | 0.369254 | 0.501477 |   0.867192 |
| LSTM     | 22.768851| 26.737422| -374.627182|

*(Consider adding key plots below by saving them to `reports/figures/` and using Markdown image syntax)*

`![XGBoost Predictions vs Actual](reports/figures/XGBoost_predictions_vs_actual.png)`
`![Random Forest Feature Importance](reports/figures/Random_Forest_Optimized_feature_importance.png)`

---

## Detailed Findings

For a comprehensive breakdown of each phase, methodology, detailed results, and discussion, please refer to the [Project Summary Report](reports/project_summary.md).

---

## Future Work

* **LSTM Improvement:** Debug and optimize the LSTM model (scaling, architecture, hyperparameters).
* **Advanced Tuning:** Perform more extensive hyperparameter tuning for XGBoost/LightGBM.
* **Feature Selection:** Apply techniques to select the most relevant features.
* **Alternative Imputation:** Explore methods for features with high initial missingness.
* **Model Ensembling:** Combine predictions from top models.
* **Deployment:** Package the best model (XGBoost) and pipeline.
* **Multi-Step Forecasting:** Extend models to predict further into the future.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
