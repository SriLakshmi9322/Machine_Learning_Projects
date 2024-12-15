from flask import Flask, request, render_template
import numpy as np
import pickle

model = pickle.load(open('LGModel', 'rb'))

# flask app
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    features = request.form['feature']
    features_lst = features.split(',')
    np_features = np.asarray(features_lst, dtype=np.float32)
    prediction = model.predict(np_features.reshape(1, -1))

    output = ["Cancerous" if prediction[0] == 1 else "Not Cancerous"]
    return render_template('index.html', message=output)


# Python Main
if __name__ == '__main__':
    app.run(debug=True)
