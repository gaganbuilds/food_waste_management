import os

from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load model (pipeline)
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
try:
    model = joblib.load(MODEL_PATH)
except Exception as exc:
    model = None
    print(f"[WARN] Unable to load model from {MODEL_PATH}: {exc}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        day_of_week = request.form['day_of_week']
        weather = request.form['weather']
        festival = int(request.form['festival'])
        expected_customers = float(request.form['expected_customers'])
        previous_day_consumption = float(request.form['previous_day_consumption'])
        previous_week_same_day = float(request.form['previous_week_same_day'])

        # Create input dataframe with original features
        input_data = pd.DataFrame({
            'Day_of_Week': [day_of_week],
            'Weather': [weather],
            'Festival': [festival],
            'Expected_Customers': [expected_customers],
            'Previous_Day_Consumption': [previous_day_consumption],
            'Previous_Week_Same_Day': [previous_week_same_day]
        })

        # Predict using the pipeline (includes preprocessing)
        prediction = model.predict(input_data)[0]
        print(f"DEBUG: Prediction result: {prediction}")  # Debug print

        return render_template('index.html', prediction=int(round(prediction)))
    except Exception as e:
        print(f"DEBUG: Error in prediction: {e}")  # Debug print
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)