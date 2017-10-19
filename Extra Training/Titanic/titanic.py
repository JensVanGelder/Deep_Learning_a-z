# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 12:09:25 2017

@author: jens
"""
# PART 1 - Importing data & libraries

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np

# Load & Combine datasets
def combine_datasets():
    #Read data
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    #Extract and remove targets from train data
    train.drop('Survived',1,inplace=True)
    #Merge data for feature engineering
    combined = train.append(test)
    combined.reset_index(inplace=True)
    combined.drop('index',inplace=True, axis=1)
    
    return combined

combined = combine_datasets()
combined.shape

# PART 2 - Data engineering

def status(feature):

    print ('Processing',feature,': ok')

# Get titles from Name
def get_titles():

    global combined
    
    # we extract the title from each name
    combined['Title'] = combined['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
    
    # a map of more aggregated titles
    Title_Dictionary = {
                        "Capt":       "Officer",
                        "Col":        "Officer",
                        "Major":      "Officer",
                        "Jonkheer":   "Royalty",
                        "Don":        "Royalty",
                        "Sir" :       "Royalty",
                        "Dr":         "Officer",
                        "Rev":        "Officer",
                        "the Countess":"Royalty",
                        "Dona":       "Royalty",
                        "Mme":        "Mrs",
                        "Mlle":       "Miss",
                        "Ms":         "Mrs",
                        "Mr" :        "Mr",
                        "Mrs" :       "Mrs",
                        "Miss" :      "Miss",
                        "Master" :    "Master",
                        "Lady" :      "Royalty"

                        }
    
    # we map each title
    combined['Title'] = combined.Title.map(Title_Dictionary)
    
get_titles()


# Will fill empty age and fare fields with mean
combined["Age"].fillna(combined.Age.median(), inplace=True)
combined["Fare"].fillna(combined.Fare.median(), inplace=True)

# Select only the numeric variables
numeric_variables = list(combined.dtypes[combined.dtypes != "object"].index)
combined[numeric_variables].head()

# Drop useless variables
combined.drop(["Name", "PassengerId"], axis=1, inplace=True)

# Change Cabin variable to be only first letter or None
def clean_cabin(x):
    try:
        return x[0]
    except TypeError:
        return "None"
combined["Cabin"] = combined.Cabin.apply(clean_cabin)

# Extrac usefull ticket info
def process_ticket():
    
    global combined
    
    # a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)
    def cleanTicket(ticket):
        ticket = ticket.replace('.','')
        ticket = ticket.replace('/','')
        ticket = ticket.split()
        ticket = map(lambda t : t.strip(), ticket)
        ticket = list(filter(lambda t : not t.isdigit(), ticket))
        if len(ticket) > 0:
            return ticket[0]
        else: 
            return 'XXX'
    

    # Extracting dummy variables from tickets:

    combined['Ticket'] = combined['Ticket'].map(cleanTicket)
    tickets_dummies = pd.get_dummies(combined['Ticket'], prefix='Ticket')
    combined = pd.concat([combined, tickets_dummies], axis=1)
    combined.drop('Ticket', inplace=True, axis=1)

    status('names')
    
process_ticket()

# Split into multiple variables
categorical_variables = ['Sex', 'Cabin', 'Embarked', 'Title']
for variable in categorical_variables:
    #Fill missing data with the word "Missing"
    combined[variable].fillna("Missing", inplace=True)
    #Create array of dummies
    dummies = pd.get_dummies(combined[variable], prefix=variable)
    #Update X to include dummies and drop the main variable
    combined = pd.concat([combined, dummies], axis=1)
    combined.drop([variable], axis=1, inplace=True)

# Recover X, y, test
def recover_train_test_target():
    global combined
    
    y = pd.read_csv('train.csv')
    y = y.Survived
    X = combined.head(891)
    test = combined.iloc[891:]
    
    return X, y, test
X, y, test = recover_train_test_target()

# PART 3 - Making the training model

# Training model
model = RandomForestRegressor(n_estimators=1000,
                              oob_score=True,
                              n_jobs=-1,
                              random_state=42,
                              max_features="auto",
                              min_samples_leaf=5)
model.fit(X,y)
roc = roc_auc_score(y, model.oob_prediction_)
print("C-stat :", roc)

# Predict test

output = model.predict(test)
output = (output > 0.5)
output=output*1
df_output = pd.DataFrame()
aux = pd.read_csv('test.csv')
df_output['PassengerId'] = aux['PassengerId']
df_output['Survived'] = output
df_output[['PassengerId','Survived']].to_csv('output.csv',index=False)

# PART 4 - Optimization ## Makes it worse for some unknown reason?

# Feature importance

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
clf = clf.fit(X, y)

# Transform X, test to more compact dataset
model = SelectFromModel(clf, prefit=True)
X_reduced = model.transform(X)
X_reduced.shape
test_reduced = model.transform(test)
test_reduced.shape

# Training model on Reduced set
model = RandomForestRegressor(n_estimators=1000,
                              oob_score=True,
                              n_jobs=-1,
                              random_state=42,
                              max_features="auto",
                              min_samples_leaf=5)
model.fit(X_reduced,y)
roc = roc_auc_score(y, model.oob_prediction_)
print("C-stat :", roc)

# Predict test_reduced
output_reduced = model.predict(test_reduced)
output_reduced = (output_reduced > 0.5)
output_reduced = output_reduced*1
df_output_reduced = pd.DataFrame()
aux = pd.read_csv('test.csv')
df_output_reduced['PassengerId'] = aux['PassengerId']
df_output_reduced['Survived'] = output_reduced
df_output_reduced[['PassengerId','Survived']].to_csv('output_reduced.csv',index=False)







