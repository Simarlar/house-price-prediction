from flask import Flask, request, render_template
import requests
import os

# Use env variable or fallback for local
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")

app = Flask(__name__)

# Home Route
@app.route('/')
def home():
    return render_template('index.html')


# Predict Route
@app.route('/predict', methods=['POST'])
def predict():

    try:
        form_data = request.form.to_dict()

        payload = {
            "Square_Footage": int(form_data["Square_Footage"]),
            "Num_Bedrooms": int(form_data["Num_Bedrooms"]),
            "Num_Bathrooms": int(form_data["Num_Bathrooms"]),
            "Year_Built": int(form_data["Year_Built"]),
            "Lot_Size": float(form_data["Lot_Size"]),
            "Garage_Size": int(form_data["Garage_Size"]),
            "Neighborhood_Quality": int(form_data["Neighborhood_Quality"])
        }

        response = requests.post(API_URL, json=payload)

        # Handle API errors
        if response.status_code != 200:
            return render_template(
                "index.html",
                prediction_text=f"Error: {response.text}"
            )

        result = response.json()

        prediction = result["prediction"]

        return render_template(
            "index.html",
            prediction_text=f"${round(prediction, 2):,}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )


if __name__ == "__main__":
    app.run(debug=True)