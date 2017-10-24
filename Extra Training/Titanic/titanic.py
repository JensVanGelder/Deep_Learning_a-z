# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 12:09:25 2017

@author: jens
"""
# PART 1 - Importing data & libraries

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score

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
    
    # Extract title
    combined['Title'] = combined['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
    
    # Map of more Aggregated Titles
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

# PART 3 - Making the training model #### NOT NECESSARY, SKIP THIS

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

# PART 4 - Optimization

# Feature importance

clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
clf = clf.fit(X, y)

# Transform X, test to more compact dataset
model = SelectFromModel(clf, prefit=True)
X_reduced = model.transform(X)
X_reduced.shape
test_reduced = model.transform(test)
test_reduced.shape

# RandomTree parameter tuning
run_gs = False

if run_gs:
    parameter_grid = {
                 'max_depth' : [4, 6, 8],
                 'n_estimators': [50, 10],
                 'max_features': ['sqrt', 'auto', 'log2'],
                 'min_samples_split': [1, 3, 10],
                 'min_samples_leaf': [1, 3, 10],
                 'bootstrap': [True, False],
                 }
    forest = RandomForestClassifier()
    cross_validation = StratifiedKFold(y, n_folds=5)

    grid_search = GridSearchCV(forest,
                               scoring='accuracy',
                               param_grid=parameter_grid,
                               cv=cross_validation)

    grid_search.fit(X, y)
    model = grid_search
    parameters = grid_search.best_params_

    print('Best score: {}'.format(grid_search.best_score_))
    print('Best parameters: {}'.format(grid_search.best_params_))
else: 
    parameters = {'bootstrap': False, 'min_samples_leaf': 3, 'n_estimators': 50, 
                  'min_samples_split': 10, 'max_features': 'sqrt', 'max_depth': 6}
    
    model = RandomForestClassifier(**parameters)
    model.fit(X, y)

def compute_score(clf, X, y, scoring='accuracy'):
    xval = cross_val_score(clf, X, y, cv = 5, scoring=scoring)
    return np.mean(xval)

compute_score(model, X, y, scoring='accuracy')


output = model.predict(test).astype(int)
df_output = pd.DataFrame()
aux = pd.read_csv('test.csv')
df_output['PassengerId'] = aux['PassengerId']
df_output['Survived'] = output
df_output[['PassengerId','Survived']].to_csv('output.csv',index=False)

