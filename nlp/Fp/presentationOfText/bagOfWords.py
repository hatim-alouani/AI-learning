import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import CountVectorizer

def tokenization_words(text):
    return word_tokenize(text)

def tokenization_sentences(text):
    return sent_tokenize(text)

def cleaning(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    return ' '.join(text)

def stop_words(tokens):
    return [token for token in tokens if token not in set(stopwords.words('english'))]

def stemming(tokens):
    return [PorterStemmer().stem(token) for token in tokens]

def lemmatization(tokens):
    return [WordNetLemmatizer().lemmatize(token, pos='v') for token in tokens]

def bow(text):
    cv = CountVectorizer()
    x = cv.fit_transform(text).toarray()
    return x

if __name__=="__main__":
    text = "NLP is a fascinating technique! I started learning it 2 days ago."
    text = cleaning(text)
    tokens = tokenization_words(text)
    tokens = stop_words(tokens)
    tokens = stemming(tokens)
    x = bow(tokens)
    print(x)
