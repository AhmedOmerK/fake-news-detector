import streamlit as st
import joblib
import string
import nltk
from nltk.corpus import stopwords

# Ensure nltk stopwords download works (even on Streamlit Cloud)
try:
    _create_unverified_https_context = ssl._create_unverified_context
    ssl._create_default_https_context = _create_unverified_https_context
except AttributeError:
    pass

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Text cleaning function (same as notebook)
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    tokens = text.split()
    tokens = [word.strip(string.punctuation) for word in tokens]
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(tokens)

# Load model + vectorizer
model = joblib.load("model/fake_news_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

# Streamlit UI
st.title("📰 Fake News Detector")
st.write("Enter a news article or headline to check if it's fake.")

user_input = st.text_area("Paste your article here:")

if st.button("Analyze"):
    cleaned = clean_text(user_input)
    vect = vectorizer.transform([cleaned])
    prediction = model.predict(vect)[0]
    confidence = model.predict_proba(vect)[0][prediction]

    if prediction == 0:
        st.error(f"⚠️ This looks like **FAKE NEWS** ({confidence:.2%} confidence)")
    else:
        st.success(f"✅ This appears to be **REAL NEWS** ({confidence:.2%} confidence)")
