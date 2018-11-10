import numpy as np
import argparse
from sklearn import svm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn import preprocessing


parser = argparse.ArgumentParser()
parser.add_argument('--C_ex', default=1, type=int)
args = parser.parse_args()
C = args.C_ex

to_save = np.zeros((45,2))

def testing(clf=None, X_train=None, y_train=None, X_test=None, y_test=None, C=None):
    #print("Testing....")
    y_pred1 = clf.predict(X_train)
    y_pred2 = clf.predict(X_test)
    y1 = accuracy_score(y_train, y_pred1)
    y2 = accuracy_score(y_test, y_pred2)
    to_save[C][0]=y1
    to_save[C][1]=y2
    return y1, y2 

def training(X_train=None, y_train=None, kernel='poly', degree=2, random_state=30, C=1):
    #print("Training for C = ", C)
    clf = SVC(C=C, gamma='auto', random_state=random_state, kernel=kernel, max_iter=100000)
    clf.fit(X_train, y_train)
    return clf

def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step

def main():

    # Read data
    data1 = pd.read_csv("spambase/spambase.data", header=None)
    data = data1.values     # Convert from pandas dataframe to numpy

    # Separate samples and labels
    X1 = data[:,:57]
    y = data[:,57]
    X1.shape

    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(X1)

    # Separate into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train for each value of C and output the prediction value in file
    for C in range(45):
        clf = training(X_train=X_train, y_train=y_train, C=np.power(2.0, C))
        pred = testing(clf=clf, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, C=C)
        #print(C-7, end=' ')
        #print(pred)
    
    np.save("results_poly_train_test", to_save)


if __name__=="__main__":
    main()