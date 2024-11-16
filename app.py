from flask import Flask, request, render_template
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

app = Flask(__name__)

# Load pre-trained model and preprocessor
with open("model/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

# Load transformed X_test and y_test for calculating metrics from CSV files
X_test = pd.read_csv("data/X_test_transformed.csv")
y_test = pd.read_csv("data/y_test.csv")["math score"]

# Make predictions on X_test to calculate metrics
y_pred = model.predict(X_test)

# Calculate model metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

@app.route("/")
def home():
    return render_template("predict.html", prediction=None, rmse=rmse, r2=r2)

@app.route("/predict", methods=["POST"])
def predict():
    # Get data from form
    gender = request.form["gender"]
    race = request.form["race"]
    parent_education = request.form["parent_education"]
    lunch = request.form["lunch"]
    test_prep = request.form["test_prep"]
    reading_score = float(request.form["reading_score"])
    writing_score = float(request.form["writing_score"])

    # Prepare input data as DataFrame
    input_data = pd.DataFrame([{
        "gender": gender,
        "race/ethnicity": race,
        "parental level of education": parent_education,
        "lunch": lunch,
        "test preparation course": test_prep,
        "reading score": reading_score,
        "writing score": writing_score
    }])

    # Transform input data using preprocessor
    transformed_data = preprocessor.transform(input_data)

    # Make prediction
    prediction = model.predict(transformed_data)[0]

    # Render the result in the form with metrics
    return render_template("predict.html", prediction=prediction, rmse=rmse, r2=r2)

if __name__ == "__main__":
    app.run(debug=True)