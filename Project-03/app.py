from flask import Flask, request, render_template
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import pickle

clf = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))

app = Flask(__name__)

stopwords_set = set(stopwords.words('english'))
emoji_pattern = re.compile('(?::|;|=)(?:-)?(?:\)|\(|D|P)')


def preprocessing(text):
    text = re.sub('<[^>]*>', '', text)
    emojis = emoji_pattern.findall(text)
    text = re.sub('[\W+]', ' ', text.lower()) + ' '.join(emojis).replace('-', '')

    porter = PorterStemmer()
    text = [porter.stem(word) for word in text.split() if word not in stopwords_set]

    return " ".join(text)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        comment = request.form['text']
        cleaned_comment = preprocessing(comment)
        comment_vector = tfidf.transform([cleaned_comment])
        prediction = clf.predict(comment_vector)[0]

        return render_template('index.html', prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
