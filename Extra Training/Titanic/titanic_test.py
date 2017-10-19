# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 10:06:33 2017

@author: jens
"""

from sklearn.ensemble import RandomForestRegressor
# Error metric, C-stat
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np

# Import data
X = pd.read_csv('train.csv')
y = X.pop("Survived")
X.describe()

# Will empty age with mean
X["Age"].fillna(X.Age.mean(), inplace=True)

# Select only the numeric variables
numeric_variables = list(X.dtypes[X.dtypes != "object"].index)
X[numeric_variables].head()

# Model
model = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=42)
model.fit(X[numeric_variables],y)

model.oob_score_

y_oob = model.oob_prediction_
print( roc_auc_score(y,y_oob))

# Function to show descriptive stats on the categorical variables
def describe_categorical(X):
    from IPython.display import display, HTML
    display(HTML(X[X.columns[X.dtypes == "object"]].describe().to_html()))
describe_categorical(X)

# Drop variables
X.drop(["Name", "Ticket", "PassengerId","Embarked_Missing","Cabin_T"], axis=1, inplace=True)
X.drop(["Embarked_Missing","Cabin_T"],axis=1, inplace=True)
# Change Cabin variable to be only first letter or None
def clean_cabin(x):
    try:
        return x[0]
    except TypeError:
        return "None"
X["Cabin"] = X.Cabin.apply(clean_cabin)

categorical_variables = ['Sex', 'Cabin', 'Embarked']
for variable in categorical_variables:
    #Fill missing data with the word "Missing"
    X[variable].fillna("Missing", inplace=True)
    #Create array of dummies
    dummies = pd.get_dummies(X[variable], prefix=variable)
    #Update X to include dummies and drop the main variable
    X = pd.concat([X, dummies], axis=1)
    X.drop([variable], axis=1, inplace=True)
    
model = RandomForestRegressor(n_estimators=100, oob_score=True, n_jobs=-1, random_state=42)
model.fit(X,y)
print( roc_auc_score(y, model.oob_prediction_))

new_prediction = model.predict(np.array([[3,26,0,0,7.9,1,0,0,0,0,0,0,0,0,1,0,0,1]]))


#Simple version to show importance of variables
model.feature_importances_
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
feature_importances.plot(kind="barh", figsize=(7,6))

#Test best n_estimator settings
results=[]
n_estimator_options = [30,50,100,200,500,1000,2000]

for trees in n_estimator_options:
    model = RandomForestRegressor(trees, oob_score=True, n_jobs=-1, random_state=42)
    model.fit(X, y)
    print(trees, "trees")
    roc = roc_auc_score(y, model.oob_prediction_)
    print("C-stat: ", roc)
    results.append(roc)
    print("")
pd.Series(results, n_estimator_options).plot()

#Test best max_features settings
results = []
max_features_options = ["auto", None, "sqrt", "log2", 0.9, 0.2]

for max_features in max_features_options:
     model = RandomForestRegressor(n_estimators=1000, oob_score=True, n_jobs=-1, random_state=42, max_features=max_features)
     model.fit(X,y)
     print (max_features, "options")
     roc = roc_auc_score(y, model.oob_prediction_)
     print("C-stat: ", roc)
     results.append(roc)
     print("")
pd.Series(results, max_features_options).plot(kind="barh", xlim=(.85,.88))

#Test best min_samples_leaf settings
results = []
min_samples_leaf_options = [1,2,3,4,5,6,7,8,9,10]
for min_samples in min_samples_leaf_options:
     model = RandomForestRegressor(n_estimators=1000,
                                   oob_score=True,
                                   n_jobs=-1,
                                   random_state=42,
                                   max_features="auto",
                                   min_samples_leaf=min_samples)
     model.fit(X,y)
     print(min_samples, "min samples")
     roc = roc_auc_score(y, model.oob_prediction_)
     print("C-stat: ", roc)
     results.append(roc)
     print("")
pd.Series(results, min_samples_leaf_options).plot()

#Final Model
model = RandomForestRegressor(n_estimators=1000,
                              oob_score=True,
                              n_jobs=-1,
                              random_state=42,
                              max_features="auto",
                              min_samples_leaf=5)
model.fit(X,y)
roc = roc_auc_score(y, model.oob_prediction_)
print("C-stat :", roc)
