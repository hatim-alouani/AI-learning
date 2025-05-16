import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report 
from sklearn.ensemble import RnadomForestClassifier

iris = load_iris()

x = iris.data 
y = iris.target 

x_treain, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) #random_state=42 is used to reproduce the results 