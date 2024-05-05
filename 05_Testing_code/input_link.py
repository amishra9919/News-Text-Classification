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

    preprocessed_text = " ".join(filtered_tokens)
    # print("preprocessed_text inside preproceeess_text funciton: ", preprocessed_text)
    return preprocessed_text

def extract_paragraphs_from_link(url):

    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')

        paragraphs = soup.find_all('p')
        # print("parass: ", paragraphs)
        if len(paragraphs) != 0:
            text = ''
            for paragraph in paragraphs:          #extracting the text form each para 
                # print("inside if ")
                text += paragraph.get_text() 
            preprocessed_text = preprocess_text(text) 
            # print("preprocessed_text: ", preprocessed_text)
        else:                                 #Failed to fetch the content"
            preprocessed_text = ''

    else:
        print("Failed to fetch the content")
        preprocessed_text = ''
    return preprocessed_text

if __name__ == "__main__":

    news_link = "https://timesofindia.indiatimes.com/city/kolkata/didi-dares-pm-narendra-modi-to-debate-her-on-basics-of-hinduism/articleshow/109851621.cms"
    paragraph = extract_paragraphs_from_link(news_link)
    print('paragraph: ', paragraph)

    if len(paragraph) == 0:
        print("Unable to fetch the content")
    else:
        # print("Paragraph: ", paragraph)
        print("inside else in model loading")
        #Loding model
        model = joblib.load(r'C:\Users\DELL\OneDrive\Desktop\major project\News-Text-Classification org\04_Trained_models\TFIDF_SVM.pkl')

        new_text = [paragraph]
        predictions = model.predict(new_text)
        print("Predicted categories for the input text:", predictions)
