import joblib

def save_model(model, filepath):
    """Save the model using joblib."""
    joblib.dump(model, filepath)

def load_model(filepath):
    """Load the model using joblib."""
    return joblib.load(filepath)