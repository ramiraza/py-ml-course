import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

get_ipython().run_line_magic("matplotlib", " inline")


# creating a range from numpy array
heart_disease = pd.read_csv('./data/heart-disease.csv')


heart_disease.head(5)


# importing and separating the data for features and labels
heart_disease = pd.read_csv("./data/heart-disease.csv")
heart_disease.head(5)


# creating X (features)
X = heart_disease.drop("target", axis=1)

# creatinge y (label) from the data
y = heart_disease["target"]
X.head()


# Choose the right model and hyper-parameters
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()

# Keeping the default hyperparameters
clf.get_params()


# Fit the model to the data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf.fit(X_train, y_train);


y_preds = clf.predict(X_test)


y_preds


y_test


# Evaluate the model on training & test data
clf.score(X_train, y_train)


clf.score(X_test, y_test)


from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

print(classification_report(y_test, y_preds))


confusion_matrix(y_test, y_preds)


accuracy_score(y_test, y_preds)


# Improve a model
# try different amount of n_estimators (hyper-parameters)
np.random.seed(42)

for i in range(10, 100, 10):
  clf = RandomForestClassifier(n_estimators=i)
  clf.fit(X_train, y_train)
  score = clf.score(X_test, y_test)
  print(f"Trying a model with {i} estimators\nModel accuracy with test set: {score * 100:.2f}get_ipython().run_line_magic("\n")", "")



# 6 save a model and load it
import pickle
pickle.dump(clf, open('random_forest_model_1.pkl', 'wb'))


# load a model
loaded_model = pickle.load(open('random_forest_model_1.pkl', 'rb'))


for i in len(loaded_model)


loaded_model.score(X_test, y_test)
