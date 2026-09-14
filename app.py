import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# 1. Download required NLTK data for the Streamlit server
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

ps = PorterStemmer()

# 2. Text Preprocessing Function
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

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

# 3. Load the freshly trained vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# 4. Build the Streamlit App UI
st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    if not input_sms.strip():
        st.warning("Please enter a message to classify.")
    else:
        # Step 1: Preprocess the text
        transformed_sms = transform_text(input_sms)
        
        # Step 2: Vectorize the text
        vector_input = tfidf.transform([transformed_sms])
        
        # Step 3: Predict using the model
        result = model.predict(vector_input)[0]
        
        # Step 4: Display the result
        if result == 1:
            st.error("🚨 This message is Spam")
        else:
            st.success("✅ This message is Not Spam")