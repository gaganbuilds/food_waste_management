import pandas as pd

def add_features(df):
    """Add engineered features to the dataframe."""
    # Weekend Indicator
    df['is_weekend'] = df['Day_of_Week'].apply(lambda x: 1 if x in ['Saturday', 'Sunday'] else 0)

    # Demand Lag Feature
    df['previous_day_ratio'] = df['Previous_Day_Consumption'] / df['Expected_Customers']

    # Weekly trend
    df['weekly_trend'] = df['Previous_Week_Same_Day'] / df['Previous_Day_Consumption']

    return df