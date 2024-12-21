import re
from flask import Flask, render_template, request
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import nltk

nltk.download('stopwords')

# Initialize Flask app
app = Flask(__name__)

# Load and preprocess data
news_df = pd.read_csv('fake_news_dataset.csv')
news_df = news_df.fillna(' ')
news_df['content'] = news_df['author'] + " " + news_df['title']

ps = PorterStemmer()


def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content


news_df['content'] = news_df['content'].apply(stemming)

# Vectorize data
X = news_df['content'].values
y = news_df['label'].values

vector = TfidfVectorizer()
vector.fit(X)
X = vector.transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)


# Define prediction function
def predict_news(input_txt):
    input_text_vectorized = vector.transform([input_txt])
    predicted_value = model.predict(input_text_vectorized)
    return predicted_value[0]


# Flask routes
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Debugging: Print the form data
        print(request.form)

        if 'news_content' in request.form:
            input_text = request.form['news_content']
            prediction = predict_news(input_text)
            result = "Fake News" if prediction == 1 else "Real News"
            return render_template('index.html', prediction_result=result, news_content=input_text)
        else:
            return render_template('index.html', prediction_result="Error: No input provided", news_content="")


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
