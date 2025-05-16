from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

iris = load_iris()

x = iris.data 
y = iris.target 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=42)

model = SVC(kernel='rbf', C=1)

model.fit(x_train, y_train)

y_predict = model.predict(x_test)
print("Accuracy: ", model.score(x_test, y_test))