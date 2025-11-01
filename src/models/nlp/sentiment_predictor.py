import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Load the model
model = joblib.load('src/models/nlp/sentiment_analysis_model.pkl')

def preprocess_text(text):
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()

        # Remove punctuation and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        # Tokenize
        tokens = word_tokenize(text)

        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]

        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]

        # Join tokens back into text
        processed_text = ' '.join(tokens)
        return processed_text
    else:
        return ''

def predict_sentiment(text):
    # Preprocess the text
    processed_text = preprocess_text(text)

    # Make prediction
    prediction = model.predict([processed_text])[0]
    sentiment = 'Positive' if prediction == 1 else 'Negative'

    # Get probability scores if available
    try:
        proba = model.predict_proba([processed_text])[0]
        confidence = proba[1] if prediction == 1 else proba[0]
    except:
        confidence = None

    return {
        'sentiment': sentiment,
        'confidence': confidence,
        'sentiment_code': int(prediction)
    }
