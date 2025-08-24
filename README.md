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
Jupyter Notebook


# Project Workflow

Exploratory Data Analysis (EDA): Understand distributions, correlations, and outliers.

Data Preprocessing: Handle missing values, feature engineering, encoding, and scaling.

Model Training: Compare different regression models (e.g., Linear Regression).

Model Evaluation: 

Prediction on Test Data: 