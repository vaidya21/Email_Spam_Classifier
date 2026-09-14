import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Page Config (Sets the browser tab title and icon)
st.set_page_config(page_title="Spam Classifier", page_icon="📧")

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# --- SIDEBAR CONTENT ---
with st.sidebar:
    st.header("About the App")
    st.write("This application uses Natural Language Processing (NLP) to detect spam messages.")
    st.divider()
    st.subheader("Model Performance")
    # Displaying your actual model metrics
    col1, col2 = st.columns(2)
    col1.metric("Accuracy", "97.1%")
    col2.metric("Precision", "100%")
    st.divider()
    st.write("Built with Python, Scikit-learn and Streamlit.")

# --- MAIN PAGE CONTENT ---
st.title("📧 Email/SMS Spam Classifier")
st.write("Paste an email or text message below to check if it is spam or legitimate.")

input_sms = st.text_area("Message Content", height=150)

if st.button('Predict Message'):
    if not input_sms.strip():
        st.warning("Please enter a message to classify.")
    else:
        with st.spinner("Analyzing message..."):
            transformed_sms = transform_text(input_sms)
            vector_input = tfidf.transform([transformed_sms])
            result = model.predict(vector_input)[0]
            
            if result == 1:
                st.error("🚨 **Spam Detected!** This message looks suspicious.")
            else:
                st.success("✅ **Not Spam!** This message looks legitimate.")
