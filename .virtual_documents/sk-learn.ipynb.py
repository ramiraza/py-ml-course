# standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic("matplotlib", " inline")


heart_disease = pd.read_csv('./data/heart-disease.csv')
heart_disease.head(5)


X = heart_disease.drop("target", axis=1)
X.head()


y = heart_disease["target"]


y.head()


# splitting the data into train/test datasets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


X_train.shape, X_test.shape


y_train.shape, y_test.shape


car_sales = pd.read_csv('./data/car-sales-extended.csv')
car_sales.head()


car_sales.dtypes


# splitting data into features/labels
X = car_sales.drop("Price", axis=1)
y = car_sales["Price"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
model.fit(X_train, y_train)


# get_dummies from pandas can encode the categorical data but it doesn't always work on numerical data
pd.get_dummies(X_train)


# importing column transformer and integer encoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

X_train.head()



one_hot = OneHotEncoder()
categorical_features = ["Make", "Colour", "Doors"]

transformer = ColumnTransformer([("one_hot", one_hot, categorical_features)], remainder="passthrough")

transformed_X = transformer.fit_transform(X)


transformed_X


X_train, X_test, y_train, y_test = train_test_split(transformed_X, y, test_size=0.2)


model.fit(X_train, y_train)
model.score(X_test, y_test)



