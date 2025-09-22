# House-Pricing
The goal of this project is to develop a machine learning model that can predict the sale price of a house based on various features. I used the 'House Prices - Advanced Regression Techniques' dataset from Kaggle.com. It can be found at https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques.

# Dataset

Training data: train.csv – contains 1,460 houses with features and sale prices.

Test data: test.csv – contains the same features but without sale prices (used for model evaluation).

# Structure


├── data/                # train.csv, test.csv
├── notebooks/           # Jupyter Notebooks for EDA, preprocessing, modeling
│           ├── 01_ExploratoryDataAnalysis.ipynb
│           ├── 02_Preprocessing.ipynb
│           └── 03_Modeling.ipynb
├── scripts/             # Python scripts with reusable functions
├── output/              # Model predictions, plots, reports
├── data_description.txt # Description of the data
├── requirements.txt     # List of required Python packages
└── README.md            # This file

# Technologies Used
Python

pandas

numpy

matplotlib

seaborn

scikit-learn

statsmodels

scipy

lightgbm

jupyter notebook


# Project Workflow

Data Preprocessing:
I handled missing values and encoding. I started by dropping duplicates. Then, it was important to analyze the number and distribution of 'NaN' values to decide how to handle them. In some columns, 'NaN' represents zero, so in these columns, 'NaN' was replaced with the string 'NA'. I did not need to use imputation.

I separated numerical and categorical features to select/encode them if necessary.

- Numerical features:
Selected by correlation matrix. I chose features with a coefficient greater than 0.45. Then, I used the Variance Inflation Factor (VIF) to exclude features that were multicollinear. I compared features with the highest VIF and excluded one of two similar features (e.g., 'GrLivArea' and 'TotRmsAbvGrd'—above grade living area square feet; 'TotalBsmtSF' and '1stFlrSF'—basement area; 'GarageCars' and 'GarageArea').

- Categorical features:
Analyzed in two ways: first, by analysis of variance (F-ANOVA test), and second, by human analysis and understanding of the features.

For the F-ANOVA test, I split the cleaned categorical features from the target and then applied the Chi-square test to exclude features that could be multicollinear.

For the human analysis (see ExploratoryDataAnalysis.ipynb), I asked myself which features I would consider important before buying a house. I followed this idea along with the features that the dataset offered me.

Exploratory Data Analysis (EDA):
I used EDA to understand distributions, correlations, and features. Features are abbreviated; I checked what they refer to and tried to understand them better to filter out useless features. This was a key step for categorical features (see data_description.txt).

Encoding:
I tried two approaches:

- Dividing between nominal and ordinal categories (nominal with OneHotEncoder, ordinal with OrdinalEncoder).

- Dummy encoding.

TODO: Outliers

The selected features have been saved in the data folder as .csv files. I built different files to be used for later modeling:

- train_cleaned_num.csv: numerical features

- train_cleaned_analyse.csv: numerical features + background-analysis selected categorical features (ordinal vs. nominal encoding)

- train_cleaned_dummy_analyse.csv: numerical features + background-analysis selected categorical features (dummy encoding)

- train_cleaned_anova.csv: numerical features + F-ANOVA selected categorical features (dummy encoding)

- whole_analyse_df.csv: numerical features + background-analysis selected categorical features (no encoding)

For modeling, I compared machine learning models applied to different feature selections.

I performed:

- Linear regression with numerical features (R²: 0.84)

- Linear regression with numerical features + background-analysis selected categorical features (ordinal vs. nominal) (R²: 0.84)

- Linear regression with numerical features + encoded background-analysis selected categorical features (R²: 0.85)

- Linear regression with numerical features + background-analysis selected categorical features encoded with OneHotEncoder (R²: 0.85)

- Linear regression with numerical features + F-ANOVA selected categorical features encoded with dummy variables (R²: 0.81)

Then, I trained a Light Gradient Boosting Machine Regressor (LGBMRegressor). It is more powerful than linear regression and is quite suitable for the House Pricing project. I obtained an R² of 0.88. LGBMRegressor with numerical features selected from the correlation matrix and categorical features selected from background analysis gave the best results.

Prediction on Test Data: see "data/test_daten_predicted_LGBMR" for the corrected results