import joblib
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "creditcard.csv")

def load_data():
    dataset = pd.read_csv(DATA_DIR)
    X = dataset[:,:-1]
    y = dataset[:-1]
    return X, y

scaler=StandardScaler()
dataset['Amount'] = scaler.fit_transform(dataset[['Amount']])
dataset['Time'] = scaler.fit_transform(dataset[['Time']])

def train (X,y,transormers=None):
    clfs = [LinearSVC(random_state=0),SVC(random_state=0),LogisticRegression(random_state=0),KNeighborsClassifier()]
    best_f1 = 0
    best_clf = None
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.2)
    for clf in clfs:
        print(f"training clf {clf}")
        if transformers:
            clf = Pipeline(transformers+[('clf', clf)])
        clf.fit(x_train, y_train)
        f1 = f1_score(y_test, clf.predict(x_test), average='micro')
        print(f"clf {clf} has f1 score of {f1}")
        if f1> best_f1:
            best_f1 = f1
            best_clf = clf
    print(f"trained with best f1 of {best_f1}")
    return best_clf



def save_model(clf):
    joblib.dump(clf, PICKLE_DIR, True)

def load_model():
    return joblib.load(PICKLE_DIR)

def run():

     # load the data
    X, Y = load_data()
    
    clf = train(X, Y)
    save_model(clf)
    print("clf trained !!")





