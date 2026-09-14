# Email & SMS Spam Classifier 📩⛔

A Machine Learning-powered web application that detects whether a given email or SMS message is "Spam" or "Not Spam". 

**[🔴 Live App: Test the Classifier Here!](https://emailspamclassifier-gb3d7uwhx6xfp4db85533d.streamlit.app/)**

## 💡 Overview
Traditional rule-based spam filters often struggle to keep up with modern, disguised spam. This project solves that by utilizing a Natural Language Processing (NLP) pipeline and a probabilistic Machine Learning model. The app processes raw text, extracts meaningful features, and instantly predicts the message class through a clean, interactive UI.

## 🚀 Model Performance
The model was trained on a dataset of over 5,000 messages and achieved excellent results on the test data:
* **Accuracy:** 97.1%
* **Precision:** 100.0% *(Zero false positives—meaning no legitimate messages were accidentally marked as spam!)*

## 🛠️ Tech Stack
* **Language:** Python
* **Frontend:** Streamlit
* **Machine Learning:** Scikit-learn (Multinomial Naive Bayes)
* **Text Vectorization:** TF-IDF (Term Frequency-Inverse Document Frequency)
* **NLP Processing:** NLTK (Tokenization, Stopwords removal, Porter Stemming)
* **Data Manipulation:** NumPy, Pandas

## 📂 Project Structure
* `app.py`: The main Python script that runs the Streamlit web interface.
* `sms-spam-detection.ipynb`: The Jupyter Notebook containing data exploration, preprocessing, model training, and evaluation.
* `model.pkl`: The saved Multinomial Naive Bayes model.
* `vectorizer.pkl`: The saved TF-IDF vectorizer (max_features=3000).
* `spam.csv`: The dataset used for training the model.
* `requirements.txt`: The list of Python dependencies required to run the app.

## 💻 How to Run Locally
If you want to run this project on your own machine, follow these steps:

**1. Clone the repository**
```
git clone https://github.com/vaidya21/Email-Spam-Classifier.git
cd Email-Spam-Classifier
```

**2. Install dependencies**
Make sure you have Python installed. Then run:
pip install -r requirements.txt

**3. Run the Streamlit app**
streamlit run app.py

*The application will open automatically in your default web browser at `http://localhost:8501/`.*
