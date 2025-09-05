# House-Pricing
The goal of this project is to develop a machine learning model that can predict the sale price of a house based on various features. I used the 'House Prices - Advanced Regression Techniques' data from Kaggle.com. It can be found at https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques.

# Dataset

Training data: train.csv – contains 1,460 houses with features and sale prices.

Test data: test.csv – contains the same features, but without sale prices (used for model evaluation).

# Structure

/
├── data/                # train.csv, test.csv
├── notebooks/           # Jupyter Notebooks for EDA, preprocessing, modeling
│   ├── 01_ExploratoryDataAnalysis.ipynb
│   ├── 02_Preprocessing.ipynb
│   └── 03_Modeling.ipynb
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
Jupyter Notebook


# Project Workflow

Exploratory Data Analysis (EDA): I used EDA to understand distributions, correlations and features. Features are abbreviated, I checked what they refer to and tried to understand them better to sort out useless fetaures. This has been a key step for categorical features (see data_description.txt)

Data Preprocessing: Handle missing values, encoding. I started with dropping duplicates. Then it was important to know the number and the distribution of the 'nan' values, in order to understand how to handle them. In some columns 'nan' represents 'zero': in these colmuns 'nan' has been substituted with a 'NA' string. I did not need to use imputation.

I separated numerical and categorical feature in order to select/ encode them if necessary.

Numerical features have been selceted by correlation matrix. I selceted the features having a coefficient greater then 0.45. Then I used a Variance Inflation Factor (VIF) to exclude the features that are multicollinear. I compared the features with the biggest VIF and I excluded one of the two that shows a similar feature ('GrLivArea', 'TotRmsAbvGrd' -> above grade (ground) living area square feet; 'TotalBsmtSF', '1stFlrSF' -> Area des Basements; 'GarageCars','GarageArea').

Categorical features has been analysed in two ways:  the first one with the anylysis of variance (or F-Anova test), the second one with human analysis and understanding of the features.

For the F-Anova test I splitted the (cleaned) categorical features from the target one, then I applied the Chi Quadrat Test to exlude features that could be multiccolinear.

For the 'human' analysis: see ExploratoryDataAnalysis.ipynb. I asked myself which fetaure I would give importance to before evaluating to buy a house. I followed this idea along the features that the dataset where offering me.

#TODO: Outliers

Model Training: Compare different regression models (Linear Regression with different features selection and different categories; LGBMRegressor with the best features selection of linear regression).

Model Evaluation: LGBMRegressor with numerical feature selection from correlation matrix and categorical feature selection from background analysis gave the best result (R²: 0.8831)

Prediction on Test Data: see "data/test_daten_predicted_LGBMR"