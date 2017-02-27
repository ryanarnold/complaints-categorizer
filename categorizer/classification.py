from sklearn import preprocessing, neighbors
import numpy as np
import pandas as pd

def get_x(csv_path):
    df = pd.read_csv(csv_path)
    # print(df['category'][0])
    # input()
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

def get_scores(actual, predicted, category):
    TP = 0
    indexes = [i for i in range(len(actual)) if actual[i] == category]
    for i in indexes:
        if actual[i] == predicted[i]:
            TP += 1

    TN = 0
    irrelevant_indexes = [i for i in range(len(actual)) if actual[i] != category]
    for i in irrelevant_indexes:
        if predicted[i] != category:
            TN += 1

    FP = 0
    for i in irrelevant_indexes:
        if predicted[i] == category:
            FP += 1

    FN = 0
    for i in indexes:
        if predicted[i] != category:
            FN += 1

    try:
        precision = TP / (TP + FP)
    except ZeroDivisionError:
        precision = 0.0
    recall = TP / (TP + FN)
    try:
        # F = '{0:.4f}'.format((precision * recall) / (precision + recall))
        F = float('{0:.4f}'.format(2 * (precision * recall) / (precision + recall)))
    except ZeroDivisionError:
        F = float('0.0')
    precision = float('{0:.4f}'.format(precision))
    recall = float('{0:.4f}'.format(recall))
    accuracy = float('{0:.4f}'.format((TP + TN) / len(actual)))

    return {'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN, 'p': precision, 'r': recall, 'F': F, 'acc': accuracy}
