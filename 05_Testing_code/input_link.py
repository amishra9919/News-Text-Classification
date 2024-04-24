import joblib



import requests
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords

def preprocess_text(text):
    text = text.lower()
    
    text = " ".join(text.split())                                                #remove spaces
    
    text = re.sub(r'[^a-zA-Z0-9\s.]', '', text)                                  #remove special symbols and characters
    
    tokens = text.split()                                                        #tokenize text by space
    
    stop_words = set(stopwords.words('english'))                                 #remove stopwords except full stop
    stop_words.discard('.')

    filtered_tokens = [token for token in tokens if token not in stop_words]     #remove stopwords

    # Join tokens back into text
    preprocessed_text = " ".join(filtered_tokens)
    
    return preprocessed_text

def extract_paragraphs_from_link(url):

    response = requests.get(url)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')

        paragraphs = soup.find_all('p')

        if len(paragraphs) != 0:
            for paragraph in paragraphs:          #extracting the text form each para 
                # print("inside if ")
                text = paragraph.get_text()
                preprocessed_text = preprocess_text(text)
        else:                                 #Failed to fetch the content"
            preprocessed_text = ''

    else:
        print("Failed to fetch the content")
        preprocessed_text = ''
    return preprocessed_text


news_link = "https://indianexpress.com/article/sports/cricket/dc-vs-gt-live-score-ipl-2024-match-40-today-delhi-capitals-vs-gujarat-titans-scorecard-updates-9287629/"
paragraph = extract_paragraphs_from_link(news_link)

if len(paragraph) == 0:
    print("Unable to fetch the content")
else:
    # print("inside else in model loading")
    #Loding model
    model = joblib.load(r'D:\VCET\BE_Project\News_Text_Classification\04_Trained_models\TFIDF_SVM.pkl')

    new_text = [paragraph]
    predictions = model.predict(new_text)
    print("Predicted categories for the input text:", predictions)
