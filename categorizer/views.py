from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.core.urlresolvers import reverse
from categorizer.preprocessing import *
from categorizer.classification import *
import numpy as np
import pandas as pd
import time
import csv

RAW_CSV_PATH = 'globals/data/raw.csv'
VECTORIZED_TRAIN_CSV_PATH = 'globals/data/vectorized_train.csv'
VECTORIZED_TEST_CSV_PATH = 'globals/data/vectorized_test.csv'
PREPROCESSED_TRAIN_JSON_PATH = 'globals/data/preprocessed_train.json'
PREPROCESSED_TEST_JSON_PATH = 'globals/data/preprocessed_test.json'
FEATURES_JSON_PATH = 'globals/data/features.json'

def index(request):
    return HttpResponseRedirect(reverse('home'))


def home(request):
    return render(request, 'home.html')


def categorizer(request):
    complaint = request.GET.get('message')

    if complaint is not None:
        # Tokenization, Stopword Removal, and Stemming
        complaint = tokenize(complaint)
        complaint = remove_stopwords(complaint)
        complaint = stem(complaint)

        complaint = {
            'id': '',
            'body': complaint,
            'category': ''
        }

        features = load_json(FEATURES_JSON_PATH)

        train_set = load_json(PREPROCESSED_TRAIN_JSON_PATH)
        test_set = load_json(PREPROCESSED_TEST_JSON_PATH)
        test_set.append(complaints)

        train_set, test_set = nb_vectorize(train_set, test_set, features)

        write_csv(train_set, VECTORIZED_TRAIN_CSV_PATH)
        write_csv(test_set, VECTORIZED_TEST_CSV_PATH)

        # Get the vectorized data, to prepare it for classification:
        df = pd.read_csv(VECTORIZED_TRAIN_CSV_PATH)

        X_train = np.array(df.drop(['category', 'id'], 1))
        y_train = np.array(df['category'])

        df = pd.read_csv(VECTORIZED_TEST_CSV_PATH)

        X_test = np.array(df.drop(['category', 'id'], 1))
        y_test = np.array(df['category'])
        id_test = np.array(df['id'])

        classifier = train_classifier(X_train, y_train)

        predict_list = X_test.reshape(len(X_test), -1)
        category_list = y_test
        predictions_num = classifier.predict(predict_list)

        cats = {
            '1': 'HR',
            '4': 'ROADS',
            '5': 'BRIDGES',
            '6': 'FLOOD CONTROL',
            '10': 'COMMENDATIONS'
        }
        category = cats[str(predictions_num[-1])]
    else:
        complaint = ''
        category = ''
        
    return render(request, 'categorizer.html', {'complaint': complaint, 'category': category})


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
        complaints = load_raw(RAW_CSV_PATH)

        # Tokenization, Stopword Removal, and Stemming
        i = 1
        for complaint in complaints:
            complaint['body'] = tokenize(complaint['body'])
            complaint['body'] = ner(complaint['body'])
            complaint['body'] = remove_stopwords(complaint['body'])
            complaint['body'] = stem(complaint['body'])
            print('Finished complaint # ' + str(i))
            i += 1

        # Partition into training set and test set
        shuffle(complaints)
        half_point = int(len(complaints) * 0.8)
        train_set = complaints[:half_point]
        test_set = complaints[half_point:]
        write_json(train_set, PREPROCESSED_TRAIN_JSON_PATH)
        write_json(test_set, PREPROCESSED_TEST_JSON_PATH)

        # Feature extraction (needed in vectorization)
        features = extract_features(train_set)
        write_json(features, FEATURES_JSON_PATH)

        # Vectorization
        train_set, test_set = nb_vectorize(train_set, test_set, features)

        # Put vectorized data in csv (sklearn reads from csv kasi)
        write_csv(train_set, VECTORIZED_TRAIN_CSV_PATH)
        write_csv(test_set, VECTORIZED_TEST_CSV_PATH)

        # Get the vectorized data, to prepare it for classification:
        train_x = get_x(VECTORIZED_TRAIN_CSV_PATH)
        train_y = get_y(VECTORIZED_TRAIN_CSV_PATH)
        test_x = get_x(VECTORIZED_TEST_CSV_PATH)
        test_y = get_y(VECTORIZED_TEST_CSV_PATH)
        test_id = get_id(VECTORIZED_TEST_CSV_PATH)
        classifier = train_classifier(train_x, train_y)

        # Prepare output for template:
        accuracy = classifier.score(test_x, test_y)
        context['accuracy'] = '{0:.4f}'.format(accuracy * 100)

        predict_list = test_x.reshape(len(test_x), -1)
        category_list = test_y
        predictions_num = classifier.predict(predict_list)
        cats = {
            '1': 'HR',
            '4': 'ROADS',
            '5': 'BRIDGES',
            '6': 'FLOOD CONTROL',
            '10': 'COMMENDATIONS'
        }
        for i in range(len(predictions_num)):
            correct = 'Yes' if predictions_num[i] == category_list[i] else 'No'
            context['prediction'].append({
                'id': test_id[i],
                'system_category': cats[str(predictions_num[i])],
                'actual_category': cats[str(category_list[i])],
                'correct': correct
            })

    return render(request, 'performance.html', context)

def traditional(request):
    context = {'accuracy': 0.0, 'prediction': [], }

    if request.method == 'POST':
        complaints = load_raw(RAW_CSV_PATH)

        # Tokenization, Stopword Removal, and Stemming
        i = 1
        for complaint in complaints:
            complaint['body'] = tokenize(complaint['body'])
            complaint['body'] = ner(complaint['body'])
            complaint['body'] = remove_stopwords(complaint['body'])
            complaint['body'] = stem(complaint['body'])
            print('Finished complaint # ' + str(i))
            i += 1

        # Partition into training set and test set
        shuffle(complaints)
        half_point = int(len(complaints) * 0.8)
        train_set = complaints[:half_point]
        test_set = complaints[half_point:]
        write_json(train_set, PREPROCESSED_TRAIN_JSON_PATH)
        write_json(test_set, PREPROCESSED_TEST_JSON_PATH)

        # Feature extraction (needed in vectorization)
        features = extract_features(train_set)
        write_json(features, FEATURES_JSON_PATH)

        # Vectorization
        train_set, test_set = nb_vectorize(train_set, test_set, features)

        # Put vectorized data in csv (sklearn reads from csv kasi)
        write_csv(train_set, VECTORIZED_TRAIN_CSV_PATH)
        write_csv(test_set, VECTORIZED_TEST_CSV_PATH)

        # Get the vectorized data, to prepare it for classification:
        train_x = get_x(VECTORIZED_TRAIN_CSV_PATH)
        train_y = get_y(VECTORIZED_TRAIN_CSV_PATH)
        test_x = get_x(VECTORIZED_TEST_CSV_PATH)
        test_y = get_y(VECTORIZED_TEST_CSV_PATH)
        test_id = get_id(VECTORIZED_TEST_CSV_PATH)
        classifier = train_classifier(train_x, train_y)

        # Prepare output for template:
        accuracy = classifier.score(test_x, test_y)
        context['accuracy'] = '{0:.4f}'.format(accuracy * 100)

        predict_list = test_x.reshape(len(test_x), -1)
        category_list = test_y
        predictions_num = classifier.predict(predict_list)
        cats = {
            '1': 'HR',
            '4': 'ROADS',
            '5': 'BRIDGES',
            '6': 'FLOOD CONTROL',
            '10': 'COMMENDATIONS'
        }
        for i in range(len(predictions_num)):
            correct = 'Yes' if predictions_num[i] == category_list[i] else 'No'
            context['prediction'].append({
                'id': test_id[i],
                'system_category': cats[str(predictions_num[i])],
                'actual_category': cats[str(category_list[i])],
                'correct': correct
            })

    return render(request, 'traditional.html', context)
