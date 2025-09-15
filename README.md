# Car Price Prediction

This repository contains two Jupyter notebooks: `car_price_pred.ipynb` and `merged_car_pred.ipynb`. The `merged_car_pred.ipynb` merges the hackathon's data with the Used Car Price Prediction Dataset original dataset as the feature distributions are close though not exactly the same. The `car_price_pred.ipynb` only uses the hackathon's dataset. The project involves data loading, exploratory data analysis (EDA), preprocessing, model training with various regressors, hyperparameter tuning using RandomizedSearchCV, model evaluation, and generating a submission file.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features](#features)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Preprocessing](#preprocessing)
- [Models Used](#models-used)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Evaluation](#evaluation)
- [Best Performing Model](#best-performing-model)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)

## Project Overview

The main goal of this project is to accurately predict car prices using a comprehensive machine learning approach. The notebook covers:
- Loading and inspecting the dataset.
- Visualizing data distributions and relationships.
- Handling missing values and categorical features.
- Training and comparing several regression models.
- Optimizing model performance through hyperparameter tuning.
- Generating a submission file with predictions on unseen test data.

## Dataset

The dataset used for this project consists of car features and their corresponding prices. It is split into `train.csv` and `test.csv` for training and evaluation.

## Features

The dataset includes features such as:
- `brand`
- `model`
- `model_year`
- `milage`
- `fuel_type`
- `engine`
- `transmission`
- `ext_col` (exterior color)
- `int_col` (interior color)
- `accident` (accident status)
- `clean_title` (clean title status)
- `price` (target variable)

## Exploratory Data Analysis (EDA)

The EDA section includes:
- Displaying the first few rows of the dataset.
- Summarizing dataset information (`.info()`).
- Providing descriptive statistics for numerical and categorical columns.
- Visualizing missing values.
- Plotting the distribution of car prices.
- Bar charts for `fuel_type` and `accident` status.
- Correlation heatmap for numerical features.

## Preprocessing

A robust preprocessing pipeline is defined, which includes:
- Replacing dashes ('-') and '–' with NaN values in specified columns.
- Imputing missing numerical values with the median.
- Scaling numerical features using `StandardScaler`.
- Handling categorical features using `OneHotEncoder` for low cardinality features and `OrdinalEncoder` for high cardinality features.

## Models Used

The following regression models are implemented and evaluated:
- Linear Regression (`LinearRegression`)
- Robust Regression (`HuberRegressor`)
- Gradient Boosting (`GradientBoostingRegressor`)
- Random Forest (`RandomForestRegressor`)
- XGBoost (`XGBRegressor`) - (Conditional on installation)
- CatBoost (`CatBoostRegressor`) - (Conditional on installation)

## Hyperparameter Tuning

`RandomizedSearchCV` is used for hyperparameter tuning of more complex models like Gradient Boosting, Random Forest, XGBoost, and CatBoost, while `GridSearchCV` is used for simpler models. Parameter grids are defined for each model to explore optimal configurations.

## Evaluation

Model performance is evaluated using 5-fold cross-validation with Root Mean Squared Error (RMSE) as the scoring metric. The cross-validation results are compared, and the best-performing model is identified.

## Best Performing Model

Based on the cross-validation results, the `CatBoost` model was identified as the overall best performer for this dataset.

## Installation

To run this notebook, you'll need the following libraries. You can install them using pip:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost catboost joblib
```

Note: `xgboost` and `catboost` are optional. The notebook will still run without them but will skip these models.

## Usage

1. Clone this repository:
   ```bash
   git clone https://github.com/millicentaumaomondi/DSN_Car_Price_Prediction.git
   cd DSN_Car_Price_Prediction
   ```
2. Place your `train.csv` and `test.csv` files in the `hackathon-qualification (1)/data/` directory (or update the data loading paths in the notebook).
3. Open and run the `car_price_pred.ipynb` notebook in a Jupyter environment.
4. The notebook will generate a `submission_overall_best_model.csv` file with predictions.

## File Structure

```
.
├── car_price_pred.ipynb
├── hackathon-qualification (1)/
│   └── data/
│       ├── train.csv
│       └── test.csv
└── README.md (this file)
```

