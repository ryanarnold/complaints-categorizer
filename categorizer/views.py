from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.core.urlresolvers import reverse
from categorizer.preprocess import *
import csv
import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

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
    context = {'accuracy': 0.0}

    if request.method == 'POST':
        complaints = []
        with open('globals/data/raw.csv', 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                complaints.append({'id': row[0], 'body': row[1], 'category': row[3]})

        print('Preprocessing complaints...')
        preprocessed_complaints = preprocess(complaints)
        print('Extracting features...')
        features = extract_features(preprocessed_complaints)
        print('Vectorizing...')
        vectorize(preprocessed_complaints, features)

        # PUT CLASSIFIER HERE
        df = pd.read_csv('globals/data/vectorized.csv')

        X = np.array(df.drop(['category'],1))
        y = np.array(df['category'])

        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

        clf = neighbors.KNeighborsClassifier()
        clf.fit(X_train, y_train)

        context['accuracy'] = clf.score(X_test, y_test)


    return render(request, 'performance.html', context)
