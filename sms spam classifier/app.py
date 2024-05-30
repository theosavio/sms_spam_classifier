import streamlit as st

import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf=pickle.load(open('vectorizers.pkl','rb'))

model=pickle.load(open('models.pkl','rb'))

#transform the text for pre-processing

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string

nltk.download('stopwords')
ps=PorterStemmer()

def transform_text(text):
  text=text.lower()
  text=nltk.word_tokenize(text)

  #removing special charcaters containing alphanumeric words

  text=[word for word in text if word.isalnum()]

  #removing stopwords and punct

  text=[word for word in text if word not in stopwords.words('english') and word not in string.punctuation]

  #applying stemming

  text=[ps.stem(word) for word in text]

  return " ".join(text)



#streamlit code

st.title("SMS spam classifier")

input_sms=st.text_area("enter messsage")

if st.button('predict'):
    # Preprocess
    transformed_sms = transform_text(input_sms)
    #st.write("Transformed Text:", transformed_sms)
    
    # Vectorize
    vector_input = tfidf.transform([transformed_sms])
    #st.write("Vectorized Input:", vector_input)
    
    # Predict
    result = model.predict(vector_input)[0]
    #st.write("Prediction Result:", result)
    
    # Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")


