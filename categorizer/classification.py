from sklearn import preprocessing, neighbors
import numpy as np
import pandas as pd

def get_x(csv_path):
    df = pd.read_csv(csv_path)
    return np.array(df.drop(['category', 'id'], 1))

def get_y(csv_path):
    df = pd.read_csv(csv_path)
    return np.array(df['category'])

def get_id(csv_path):
    df = pd.read_csv(csv_path)
    return np.array(df['id'])

def train_classifier(train_x, train_y):
    clf = neighbors.KNeighborsClassifier()
    clf.fit(train_x, train_y)

    return clf