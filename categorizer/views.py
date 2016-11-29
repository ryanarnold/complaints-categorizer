from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.core.urlresolvers import reverse
from categorizer.preprocess import *
import csv
import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd
import time
import json
from random import shuffle
from nltk.classify import accuracy
from nltk import word_tokenize, PorterStemmer, FreqDist, NaiveBayesClassifier

def index(request):
    return HttpResponseRedirect(reverse('home'))

def home(request):
    return render(request, 'home.html')

def categorizer(request):
    return render(request, 'categorizer.html')

def categorized(request):
    return render(request, 'categorized.html')

def complaints(request):
    return render(request, 'complaints.html')
    
def adminpage(request):
    return render(request, 'adminpage.html')
    
def data(request):
    return render(request, 'data.html')
    
def performance(request):
    context = {'accuracy': 0.0, 'prediction': [], }

    if request.method == 'POST':
        complaints = []
        with open('globals/data/raw.csv', 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                complaints.append({'id': row[0], 'body': row[1], 'category': row[3]})

        # Preprocessing:
        start = time.time()
        print('Preprocessing complaints...')
        preprocessed_complaints = preprocess(complaints)
        print('Finished after {0:.4f} seconds.'.format(time.time() - start))

        # Feature Extraction:
        start = time.time()
        print('Extracting features...')
        # features = extract_features(preprocessed_complaints)
        # with open('globals/data/features.json', 'w') as features_file:
        #     json.dump(features, features_file)
        with open('globals/data/features.json', 'r') as features_file:
            features = json.load(features_file)
        print('Finished after {0:.4f} seconds.'.format(time.time() - start))

        # Partition into training set and test set
        shuffle(preprocessed_complaints)
        half_point = int(len(complaints) * 0.8)
        train_set = preprocessed_complaints[:half_point]
        test_set = preprocessed_complaints[half_point:]

        # Vectorization:
        start = time.time()
        print('Vectorizing...')
        # vectorize(preprocessed_complaints, features)
        vectorized_train_set, vectorized_test_set = nb_vectorize(train_set, test_set, features)

        with open('globals/data/train.json', 'w') as file:
            json.dump(vectorized_train_set, file)
        with open('globals/data/test.json', 'w') as file:
            json.dump(vectorized_test_set, file)

        with open('globals/data/train.json', 'r') as file:
            vectorized_train_set = json.load(file)
        with open('globals/data/test.json', 'r') as file:
            vectorized_test_set = json.load(file)

        with open('globals/data/vectorized_train.csv', 'w') as file:
            file.write('id,')
            for category in vectorized_train_set[0]['vector'].keys():
                file.write(category + ',')
            file.write('category\n')
            for complaint in vectorized_train_set:
                file.write(complaint['id'] + ',')
                for category in vectorized_train_set[0]['vector'].keys():
                    file.write(str(complaint['vector'][category]) + ',')
                file.write(complaint['category'] + '\n')

        with open('globals/data/vectorized_test.csv', 'w') as file:
            file.write('id,')
            for category in vectorized_test_set[0]['vector'].keys():
                file.write(category + ',')
            file.write('category\n')
            for complaint in vectorized_test_set:
                file.write(complaint['id'] + ',')
                for category in vectorized_test_set[0]['vector'].keys():
                    file.write(str(complaint['vector'][category]) + ',')
                file.write(complaint['category'] + '\n')

        print('Finished after {0:.4f} seconds.'.format(time.time() - start))

        # Classification:
        start = time.time()
        print('Training...')
        df = pd.read_csv('globals/data/vectorized_train.csv')

        X_train = np.array(df.drop(['category', 'id'],1))
        y_train = np.array(df['category'])

        df = pd.read_csv('globals/data/vectorized_test.csv')

        X_test = np.array(df.drop(['category', 'id'],1))
        y_test = np.array(df['category'])
        id_test = np.array(df['id'])

        # X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

        clf = neighbors.KNeighborsClassifier()
        clf.fit(X_train, y_train)

        # Prepare output:
        context['accuracy'] = '{0:.4f}'.format(clf.score(X_test, y_test) * 100)
        print('Finished after {0:.4f} seconds.'.format(time.time() - start))

        predict_list = X_test.reshape(len(X_test), -1)
        category_list = y_test
        predictions_num = clf.predict(predict_list)
        cats = {
            '1': 'HR',
            '4': 'ROADS',
            '5': 'BRIDGES',
            '6': 'FLOOD CONTROL'
        }
        for i in range(len(predictions_num)):
            correct = 'Yes' if predictions_num[i] == category_list[i] else 'No'    
            context['prediction'].append({'id': id_test[i], 'system_category': cats[str(predictions_num[i])], 'actual_category': cats[str(category_list[i])], 'correct': correct})
        

    return render(request, 'performance.html', context)
