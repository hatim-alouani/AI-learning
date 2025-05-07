import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

dataset = pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:,3:-1].values
y = dataset.iloc[:,-1].values

le = LabelEncoder()
x[:,2] = le.fit_transform(x[:,2])

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=0)

ann = tf.keras.models.Sequential()

ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) #for the binary classification 
#ann.compile(optimizer='adam', loss='category_crossentropy', metrics=['accuracy']) #for the non binary classification

ann.fit(x_train, y_train, batch_size=32, epochs=100)

pred = ann.predict(sc.transform([[1,0,0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5

print(pred)

y_pred = ann.predict(x_test) > 0.5

print(y_pred)
print(y_test)

accuracy_score(y_test, y_pred)