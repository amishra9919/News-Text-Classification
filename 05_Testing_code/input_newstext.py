import os
import re
import joblib
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, '..', '04_Trained_models', 'TFIDF_SVM.pkl')
model = joblib.load(model_path) 

def preprocess_text(text):
    text = text.lower()
    
    text = re.sub(r'\s+', ' ', text)
    
    text = re.sub(r'[^a-zA-Z0-9\s.]', '', text)
    
    words = word_tokenize(text)
    
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words or word == '.']
    
    preprocessed_text = ' '.join(filtered_words)
    
    return preprocessed_text

news_text = "It's yet another doubleheader day in the Indian Premier League 2024 season. Kolkata Knight Riders will be up against Royal Challengers Bengaluru in the first game of the day whereas Punjab Kings will take the field against Gujarat Titans in the evening clash. On the other hand, the final of the Barcelona Open will also be contested today. All of that and much more in today's sports wrap."

# Preprocess the input text
preprocessed_text = preprocess_text(news_text)

# Make predictions
predicted_category = model.predict([preprocessed_text])

print("Predicted category:", predicted_category)
