import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data(filepath):
    """Load the dataset from CSV file."""
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    """Preprocess the data: handle missing values, encode categoricals, create pipeline."""
    # Handle missing values
    df = df.dropna()

    # Define features
    categorical_features = ['Day_of_Week', 'Weather']
    numerical_features = ['Festival', 'Expected_Customers', 'Previous_Day_Consumption', 'Previous_Week_Same_Day']

    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), categorical_features),
            ('num', StandardScaler(), numerical_features)
        ])

    X = df.drop('Meals_Consumed', axis=1)
    y = df['Meals_Consumed']

    return X, y, preprocessor