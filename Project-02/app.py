from flask import Flask, request, render_template
import numpy as np
import pickle

# Loading Models
dtr = pickle.load(open('dtr.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))

# Creating Flask app
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        year = request.form['Year']
        average_rain_fall_mm_per_year = request.form['average_rain_fall_mm_per_year']
        pesticides_tonnes = request.form['pesticides_tonnes']
        avg_temp = request.form['avg_temp']
        area = request.form['Area']
        item = request.form['Item']

        features = np.array([[year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, area, item]])

        transformed_features = preprocessor.transform(features)
        predicted_value = dtr.predict(transformed_features).reshape(1, -1)

        return render_template('index.html', predicted_value=predicted_value)


# Python Main
if __name__ == '__main__':
    app.run(debug=True)
