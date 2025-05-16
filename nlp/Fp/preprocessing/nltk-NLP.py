import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer , PorterStemmer
from nltk.corpus import stopwords
import re

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

if __name__ == "__main__":
    nltk.download('punkt')
    nltk.download('wordnet') # is for lemmatization
    nltk.download('stopwords')

    text = "The cats are playing with the mice. The mice are scared of the cats, but the cats are not scared of the mice."

    print("\n Original text:")
    print(text)

    print("\n Sentence Tokenization:")
    print(tokenization_sentences(text))


    print("\n Cleaned text:")
    cleaned_text = cleaning(text)
    print(cleaned_text)

    print("\n Word Tokenization:")
    tokens = tokenization_words(cleaned_text)
    print(tokens)

    print("\n Lemmatization:")
    lemmes = lemmatization(tokens)
    print(lemmes)

    print("\n Stemming:")
    stems = stemming(tokens)
    print(stems)

    print("\n Tokenization After removing stopwords:")
    print(stop_words(stems))