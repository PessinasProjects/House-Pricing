# House-Pricing
The goal of this project is to develop a machine learning model that can predict the sale price of a house based on various features.

# Dataset

Training data: train.csv – contains 1,460 houses with features and sale prices.

Test data: test.csv – contains the same features, but without sale prices (used for model evaluation).

# Structure

/
├── data/                # train.csv, test.csv
├── notebooks/           # Jupyter Notebooks for EDA, preprocessing, modeling
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   └── 03_Modeling.ipynb
├── scripts/             # Python scripts with reusable functions
├── output/              # Model predictions, plots, reports
├── data_description.txt # Description of the data
├── requirements.txt     # List of required Python packages
└── README.md            # This file

# Technologies Used
Python 3.x
pandas, numpy
matplotlib, seaborn
scikit-learn
statsmodels
Jupyter Notebook


# Project Workflow

Exploratory Data Analysis (EDA): Understand distributions, correlations, and outliers.

Data Preprocessing: Handle missing values, feature engineering, encoding.

Model Training: Compare different regression models (Linear Regression with different features selection and different categories; LGBMRegressor with the best features selection of linear regression).

Model Evaluation: LGBMRegressor with numerical feature selection from correlation matrix and categorical feature selection from background analysis gave the best result (R²: 0.8754)

Prediction on Test Data: see "data/test_daten_predicted_LGBMR"