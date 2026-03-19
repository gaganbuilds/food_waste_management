from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

def train_models(X_train, y_train, preprocessor):
    """Train and tune models."""
    # Define pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', RandomForestRegressor(random_state=42))])

    # Hyperparameter tuning
    param_grid = {
        'model__n_estimators': [100, 200],
        'model__max_depth': [10, 20, None]
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_absolute_error')
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    return best_model