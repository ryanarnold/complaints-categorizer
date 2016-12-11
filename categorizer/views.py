from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.core.urlresolvers import reverse
from categorizer.preprocessing import *
from categorizer.classification import *
import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd
import time
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

        execute_preprocessing(complaints)

        # Get the vectorized data, to prepare it for classification:
        start = time.time()
        print('Training...')
        df = pd.read_csv('globals/data/vectorized_train.csv')

        X_train = np.array(df.drop(['category', 'id'],1))
        y_train = np.array(df['category'])

        df = pd.read_csv('globals/data/vectorized_test.csv')

        X_test = np.array(df.drop(['category', 'id'],1))
        y_test = np.array(df['category'])
        id_test = np.array(df['id'])

        classifier = train_classifier(X_train, y_train)

        # Prepare output:
        context['accuracy'] = '{0:.4f}'.format(classifier.score(X_test, y_test) * 100)
        print('Finished after {0:.4f} seconds.'.format(time.time() - start))

        predict_list = X_test.reshape(len(X_test), -1)
        category_list = y_test
        predictions_num = classifier.predict(predict_list)
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
