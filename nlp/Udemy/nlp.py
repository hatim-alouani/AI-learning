import pandas as pd
import re # Regular expression library
import nltk # Natural Language Toolkit
nltk.download('stopwords') #downloading the stopwords
from nltk.corpus import stopwords # Importing the stopwords
from nltk.stem.porter import PorterStemmer # Importing the stemmer , stemmer is used to reduce words to their root form
from sklearn.feature_extraction.text import CountVectorizer # Importing the CountVectorizer 

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3) #quoting=3 to ignore double quotes

# Cleaning the texts
corpus = []
for i in range(0, 1000):
  review = re.sub('[^a-zA-Z]',' ', dataset['Review'][i])
  review = review.lower()
  review = review.split()
  ps = PorterStemmer()
  review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
  review = ' '.join(review)
  corpus.append(review)

print(corpus[:5])

# Creating the Bag of Words model
cv = CountVectorizer(max_features=1500)
x = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,-1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#accuracy
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)
print(f'Accuracy: {acc*100:.2f}%')