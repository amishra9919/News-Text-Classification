import os
import joblib
import requests
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords

def preprocess_text(text):
    text = text.lower()
    text = " ".join(text.split())
    text = re.sub(r'[^a-zA-Z0-9\s.]', '', text)
    tokens = text.split()
    
    stop_words = set(stopwords.words('english'))
    stop_words.discard('.')
    filtered_tokens = [token for token in tokens if token not in stop_words]
    preprocessed_text = " ".join(filtered_tokens)
    
    return preprocessed_text

def extract_paragraphs_from_link(url):
    response = requests.get(url)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        
        if len(paragraphs) != 0:
            preprocessed_text = ''
            for paragraph in paragraphs:
                text = paragraph.get_text()
                preprocessed_text += preprocess_text(text) + ' '
        else:
            preprocessed_text = ''
    else:
        print("Failed to fetch the content")
        preprocessed_text = ''
    
    return preprocessed_text

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, '..', '04_Trained_models', 'TFIDF_SVM.pkl')

news_link = "https://www.bbc.com/sport/cricket/articles/c1d4jyk300do"
paragraph = extract_paragraphs_from_link(news_link)

if len(paragraph) == 0:
    print("Unable to fetch the content")
else:
    # print(paragraph)
    #loading model
    model = joblib.load(model_path) 

    new_text = [paragraph]
    predictions = model.predict(new_text)
    print("Predicted categories for the input text: ", predictions[0])
