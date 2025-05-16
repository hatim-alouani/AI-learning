from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

iris = load_iris()

x = iris.data 
y = iris.target 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) #random_state=42 is used to reproduce the results
model = RandomForestClassifier(n_estimators=100, random_state=42) #n_estimators=100 is the number of trees in the forest 
model.fit(x_train, y_train) #fit the model to the training data
y_pred = model.predict(x_test) #predict the test data
print("Accuracy: ", accuracy_score(y_test, y_pred)) #print the accuracy of the model
