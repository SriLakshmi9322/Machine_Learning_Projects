from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
from PIL import Image
import os

# Initialize Flask app
app = Flask(__name__)


# Load the trained model from the pickle file
with open('RegModel.pkl', 'rb') as file:
    reg = pickle.load(file)


# Load dataset for displaying sample data
gold_data = pd.read_csv('gold_price_data.csv')


# Route for the main page
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    user_input = None

    if request.method == 'POST':
        # Get input data from the form
        user_input = request.form.get('features')
        try:
            # Convert input string to a numpy array
            input_array = np.array([float(x) for x in user_input.split(',')]).reshape(1, -1)

            # Make prediction using the loaded model
            prediction = reg.predict(input_array)[0]
        except ValueError:
            prediction = "Invalid input. Please provide numeric values separated by commas."

    # Convert data to HTML table for rendering
    data_html = gold_data.to_html(classes='table table-striped', index=False)
    return render_template(
        'index.html',
        data=data_html,
        prediction=prediction,
        user_input=user_input,
        image_url='/static/Golden-chart-2.jpg'
    )


# Main block
if __name__ == '__main__':
    # Ensure static files path is set for Flask
    os.makedirs('static', exist_ok=True)
    # Copy the gold.jpg image into the static folder
    img = Image.open('gold.jpg')
    img.save('static/gold.jpg')

    app.run(debug=True)
