import spacy

def tokenization_infos(text):
    for token in text:
        print(token.text)
        print(token.shape)
        print(token.is_alpha)
        print(token.is_stop)
        print(token.like_url)
        print(token.like_email)
        print('-------------')

def tokenization_words(text):
    for token in text:
        print(token.text, end=' | ')


def tokenization_sentences(text):
    return [token.text for token in text.sents]

def lemmatization(text):
    for token in text:
        print(token.text, '\t', token.lemma_)

def grammatical_role(text):
    for token in text:
       print(token.text, '\t', token.pos_) #It helps identify the grammatical role of each word in a sentence

def stop_words(text):
    tokens = [token.text for token in text if not token.is_stop]
    return tokens

def cleaning(text):
    tokens = [token.text for token in text if token.is_alpha]
    text = ' '.join(tokens)
    return english(text)

if __name__=="__main__":
    english = spacy.load("en_core_web_sm") #english library
    text = english('NLP is a fascinating technique! I started learning it 2 days ago.')
    print("Original text:")
    print(text)

    print('\n Sentence Tokenization:')
    tokenization_sentences(text)

    print('\n Word Tokenization:')
    tokenization_words(text)

    print('\n lemmatization :')
    lemmatization(text)

    print('\n grammatical role:')
    grammatical_role(text)

    print("\n Cleaned text:")
    text = cleaning(text)
    print(text)

    print("\n Tokenization After removing stopwords:")
    tokens = stop_words(text)
    print(tokens)





