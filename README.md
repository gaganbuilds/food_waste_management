# Smart Food Waste Prediction System

## Project Overview

This project builds a machine learning system to predict the number of meals that should be prepared for a given day, helping to reduce food waste by optimizing food preparation based on historical consumption data.

## Problem Statement

Food waste is a significant issue in the hospitality industry. Over-preparing meals leads to waste, while under-preparing can result in customer dissatisfaction. This system uses historical data to predict optimal meal preparation quantities.

## Dataset Description

The dataset contains the following columns:
- Day_of_Week: Day of the week (categorical)
- Festival: Binary indicator for festival days
- Weather: Weather condition (categorical)
- Expected_Customers: Number of expected customers
- Previous_Day_Consumption: Meals consumed the previous day
- Previous_Week_Same_Day: Meals consumed on the same day last week
- Meals_Consumed: Target variable (number of meals consumed)

## Machine Learning Pipeline

1. Data Loading and Exploration
2. Data Preprocessing (encoding, scaling)
3. Feature Engineering (weekend indicator, ratios)
4. Model Training (RandomForest, GradientBoosting, XGBoost)
5. Model Evaluation and Selection
6. Model Deployment via Flask Web App

## Model Performance

- MAE: [To be updated after training]
- RMSE: [To be updated]
- R² Score: [To be updated]

## Flask Deployment

The model is deployed as a web application using Flask, providing a user-friendly interface for real-time predictions.

## How to Run the Project

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the training notebook: `jupyter notebook notebooks/model_training.ipynb`
4. Start the Flask app: `python app.py`
5. Open browser to `http://localhost:5000`

## Project Structure

```
smart_food_waste_prediction/
├── app.py
├── requirements.txt
├── README.md
├── model.pkl
├── encoder.pkl
├── data/
│   └── dataset.csv
├── notebooks/
│   └── model_training.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── utils.py
├── templates/
│   └── index.html
└── static/
    └── style.css
```# LD_ML_4
# LD_ML_4
# LD_ML_4
# LD_ML_4
