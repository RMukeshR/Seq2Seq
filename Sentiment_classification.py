import numpy as np
import pandas as pd
import re
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

print("ok")

imdb_data = pd.read_table("dataset/imdb_labelled.txt", names = ['sentence', 'label'])
amazon_data = pd.read_table("dataset/amazon_cells_labelled.txt", names = ['sentence', 'label'])
yelp_data = pd.read_table("dataset/yelp_labelled.txt", names = ['sentence', 'label'])

data = pd.concat([imdb_data,amazon_data,yelp_data])
print(data.head(5))

def preprocess_text(text):
    # Lowercasing
    text = text.lower()

    # Removing special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')


    # Removing special characters and numbers
    pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    i = pattern.sub('', i)
    i = re.sub(r"[,.\"!@#$%^&*(){}?/;`~:<>+=-]", "", i)

    # Tokenization
    tokens = word_tokenize(text)

    # Removing stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in stemmed_tokens]

    # Rejoin the tokens into a preprocessed text
    preprocessed_text = ' '.join(lemmatized_tokens)

    return preprocessed_text
def clean(data):
    review = data["sentence"].tolist()
    cleaned_data=[]

    for i in review:
        
        #lower
        i = i.lower()


        # Removing special characters and numbers
        pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        i = pattern.sub('', i)
        i = re.sub(r"[,.\"!@#$%^&*(){}?/;`~:<>+=-]", "", i)
        
        #tockenize
        tokens = word_tokenize(i)

        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]
        words = [word for word in stripped if word.isalpha()]
        stop_words = set(stopwords.words("english"))
        stop_words.discard("not")
        PS = PorterStemmer()
#         words = [w for w in words if not w in stop_words]
        words = [PS.stem(w) for w in words if not w in stop_words]
        words = ' '.join(words)
        cleaned_data.append(words)
    return cleaned_data


clean_data = clean(data)


print(data[:5])
print(clean_data[:5])