from django.shortcuts import render
from django.http import HttpResponseRedirect
from django.core.urlresolvers import reverse
from categorizer.preprocessing import *
from categorizer.classification import *
import numpy as np
import pandas as pd
import time
import csv
from io import TextIOWrapper
from categorizer.forms import *
from categorizer.models import *

RAW_CSV_PATH = 'globals/data/raw.csv'
LOAD_PATH = 'globals/data/'
VECTORIZED_TRAIN_CSV_PATH = 'globals/data/vectorized_train.csv'
VECTORIZED_TEST_CSV_PATH = 'globals/data/vectorized_test.csv'
VECTORIZED_TRAIN_INPUT_CSV_PATH = 'globals/data/vectorized_train_input.csv'
VECTORIZED_TEST_INPUT_CSV_PATH = 'globals/data/vectorized_test_input.csv'
PREPROCESSED_TRAIN_JSON_PATH = 'globals/data/preprocessed_train.json'
PREPROCESSED_TEST_JSON_PATH = 'globals/data/preprocessed_test.json'
FEATURES_JSON_PATH = 'globals/data/features.json'

do_preprocessing = False

CATEGORIES = {
    '1': 'HR',
    '4': 'ROADS',
    '5': 'BRIDGES',
    '6': 'FLOOD CONTROL',
    '10': 'COMMENDATIONS'
}

SUBCATEGORIES = {
    '1': 'EMPLOYMENT',
    '2': 'PAYMENT OF SALARIES',
    '3': 'ALLEGATION OF MISBEHAVIOR/MALFEASANCE',
    '5': 'CLAIMS OF BENEFITS',
    '6': 'ALLEGATION OF DEFECTIVE ROAD CONSTRUCTION',
    '7': 'ALLEGATION PF DELAYED ROAD CONSTRUCTION',
    '8': 'ROAD SAFETY',
    '9': 'ROAD SIGNS',
    '10': 'POOR ROAD CONDITION',
    '11': 'REQUEST FOR FUNDING',
    '13': 'POOR BRIDGE CONDITION',
    '14': 'BRIDGE SAFETY',
    '15': 'ALLEGATION OF DEFECTIVE BRIDGE CONSTRUCTION',
    '16': 'ALLEGATION OF DELAYED BRIDGE CONSTRUCTION',
    '21': 'CLOOGED DRAINAGE',
    '22': 'DEFECTIVE FLOOD CONTROL CONSTRUCTION',
    '23': 'FLOOD CONTROL SAFETY',
    '24': 'REQUEST FOR FUNDING',
    '25': 'DELAYED FLOOD CONTROL CONSTRUCTION',
    '26': 'APPLICATION',
    '27': 'REQUEST FOR FUNDING'
}

def index(request):
    return HttpResponseRedirect(reverse('home'))


def home(request):
    return render(request, 'home.html')


def categorizer(request):
    complaint = request.GET.get('message')
    complaintmessage = complaint

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
        test_set.append(complaint)

        train_set, test_set = nb_vectorize(train_set, test_set, features, CATEGORIES.keys())

        write_csv(train_set, VECTORIZED_TRAIN_INPUT_CSV_PATH)
        write_csv(test_set, VECTORIZED_TEST_INPUT_CSV_PATH)

        # Get the vectorized data, to prepare it for classification:
        df = pd.read_csv(VECTORIZED_TRAIN_INPUT_CSV_PATH)

        X_train = np.array(df.drop(['category', 'id'], 1))
        y_train = np.array(df['category'])

        df = pd.read_csv(VECTORIZED_TEST_INPUT_CSV_PATH)

        X_test = np.array(df.drop(['category', 'id'], 1))
        y_test = np.array(df['category'])
        id_test = np.array(df['id'])

        classifier = train_classifier(X_train, y_train)

        predict_list = X_test.reshape(len(X_test), -1)
        category_list = y_test
        predictions_num = classifier.predict(predict_list)

        category = CATEGORIES[str(predictions_num[-1])]
    else:
        complaint = ''
        category = ''
        
    return render(request, 'categorizer.html', {'complaint': complaintmessage, 'category': category})

def multicategorizer(request):
    context = {
        'accuracy': 0.0, 
        'prediction': []
    }
    complaints = []
    complaintpath = ""
    saved = False
    if request.method == 'POST':
        # fileform = JobFileSubmitForm(request.POST, request.FILES)
        # print(fileform)
        # if fileform.is_valid():
        #     print('hey')
        #     jfs = fileform.save(commit=True)
        #     file = request.FILES['file']
        #     jfs.file = file.name
        #     jfs.uploadDate = datetime.now()
        #     # Save to DB
        #     jfs.save()
            #complaints = load_raw(RAW_CSV_PATH)
        complaint = request.FILES['csvfile'].name
        complaintpath = LOAD_PATH + complaint
        print (complaint)
        #reader = csv.DictReader(complaint)
        print (LOAD_PATH)
        print(complaint)
        print(complaintpath)
        inputcomplaints = load_multi(complaintpath)
        #with open(complaintpath, 'r', encoding='utf-8') as file:
        #    reader = csv.reader(file)
       #     for row in reader:
        #        complaints.append({
        #            'id': row[0],
        #            'body': row[1],
        #            'category': row[3]
        #        })
        #path = complaint.temporary_file_path
        #print (path)


        #print (complaints)
        # Tokenization, Stopword Removal, and Stemming

        complaints = load_raw(RAW_CSV_PATH)
        i = 1
        for complaint in complaints:
            complaint['body'] = tokenize(complaint['body'])
            #complaint['body'] = ner(complaint['body'])
            complaint['body'] = remove_stopwords(complaint['body'])
            complaint['body'] = stem(complaint['body'])
            #print('Finished complaint # ' + str(i))
            i += 1

        for complaint in inputcomplaints:
            complaint['body'] = tokenize(complaint['body'])
            #complaint['body'] = ner(complaint['body'])
            complaint['body'] = remove_stopwords(complaint['body'])
            complaint['body'] = stem(complaint['body'])
            print('Finished complaint # ' + str(i))
            i += 1

        # Partition into training set and test set

        shuffle(complaints)
        half_point = int(len(complaints) * 0.8)
        train_set = complaints[:half_point]
        test_set = inputcomplaints
        write_json(train_set, PREPROCESSED_TRAIN_JSON_PATH)
        write_json(test_set, PREPROCESSED_TEST_JSON_PATH)

        # Feature extraction (needed in vectorization)
        features = extract_features(train_set, CATEGORIES.keys())
        write_json(features, FEATURES_JSON_PATH)

        # Vectorization
        train_set, test_set = nb_vectorize(train_set, test_set, features, CATEGORIES.keys())

        # Put vectorized data in csv (sklearn reads from csv kasi)
        write_csv(train_set, VECTORIZED_TRAIN_CSV_PATH)
        write_csv(test_set, VECTORIZED_TEST_CSV_PATH)

        # Get the vectorized data, to prepare it for classification:
        df = pd.read_csv(VECTORIZED_TRAIN_CSV_PATH)

        X_train = np.array(df.drop(['category', 'id'], 1))
        y_train = np.array(df['category'])

        df = pd.read_csv(VECTORIZED_TEST_CSV_PATH)

        X_test = np.array(df.drop(['category', 'id'], 1))
        y_test = np.array(df['category'])
        test_id = get_id(VECTORIZED_TEST_CSV_PATH)
        classifier = train_classifier(X_train, y_train)


        # Prepare output for template:

        predict_list = X_test.reshape(len(X_test), -1)
        category_list = y_test
        predictions_num = classifier.predict(predict_list)

        # FOR SUBCATEGORY
        complaints = load_raw1(RAW_CSV_PATH)

        # Tokenization, Stopword Removal, and Stemming
        i = 1
        for complaint in complaints:
            complaint['body'] = tokenize(complaint['body'])
            #complaint['body'] = ner(complaint['body'])
            complaint['body'] = remove_stopwords(complaint['body'])
            complaint['body'] = stem(complaint['body'])
            print('Finished complaint # ' + str(i))
            i += 1

        # Partition into training set and test set
        shuffle(complaints)
        half_point = int(len(complaints) * 0.8)
        train_set = complaints[:half_point]
        test_set = inputcomplaints
        write_json(train_set, PREPROCESSED_TRAIN_JSON_PATH)
        write_json(test_set, PREPROCESSED_TEST_JSON_PATH)

        # Feature extraction (needed in vectorization)
        features = extract_features(train_set, SUBCATEGORIES.keys())
        write_json(features, FEATURES_JSON_PATH)

        # Vectorization
        train_set, test_set = nb_vectorize(train_set, test_set, features, SUBCATEGORIES.keys())

        # Put vectorized data in csv (sklearn reads from csv kasi)
        write_csv(train_set, VECTORIZED_TRAIN_CSV_PATH)
        write_csv(test_set, VECTORIZED_TEST_CSV_PATH)

        # Get the vectorized data, to prepare it for classification:
        df = pd.read_csv(VECTORIZED_TRAIN_CSV_PATH)

        X_train = np.array(df.drop(['category', 'id'], 1))
        y_train = np.array(df['category'])

        df = pd.read_csv(VECTORIZED_TEST_CSV_PATH)

        X_test = np.array(df.drop(['category', 'id'], 1))
        y_test = np.array(df['category'])
        test_id = np.array(df['id'])
        classifier = train_classifier(X_train, y_train)


        # Prepare output for template:

        predict_list = X_test.reshape(len(X_test), -1)
        category_list = y_test
        predictions_subnum = classifier.predict(predict_list)

        for i in range(len(predictions_num)):
            context['prediction'].append({
                'id': test_id[i],
                'body': inputcomplaints[i]['body'],
                'system_category': CATEGORIES[str(predictions_num[i])],
                'system_subcategory': SUBCATEGORIES[str(predictions_subnum[i])]
            })

        filepath = 'globals/static/report.csv'
        with open(filepath, 'w') as file:
            for c in context['prediction']:
                file.write(c['id'])
                file.write(',')
                file.write(c['body'])
                file.write(',')
                file.write(c['system_category'])
                file.write(',')
                file.write(c['system_subcategory'])
                file.write('\n')

    return render(request, 'multicategorizer.html', context)

def performance(request):
    context = {
        'accuracy': 0.0, 
        'prediction': [], 
        'road_scores': {'TP': '','TN': '','FP': '','FN': '','p':'','r':'','F':'','acc':''},
        'hr_scores': {'TP': '','TN': '','FP': '','FN': '','p':'','r':'','F':'','acc':''},
        'flood_scores': {'TP': '','TN': '','FP': '','FN': '','p':'','r':'','F':'','acc':''},
        'commend_scores': {'TP': '','TN': '','FP': '','FN': '','p':'','r':'','F':'','acc':''},
        'bridge_scores': {'TP': '','TN': '','FP': '','FN': '','p':'','r':'','F':'','acc':''},
    }

    if request.method == 'POST':
        if do_preprocessing:
            complaints = load_raw(RAW_CSV_PATH)

            # Tokenization, Stopword Removal, and Stemming
            i = 1
            for complaint in complaints:
                complaint['body'] = tokenize(complaint['body'])
                #complaint['body'] = ner(complaint['body'])
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
            features = extract_features(train_set, CATEGORIES.keys())
            write_json(features, FEATURES_JSON_PATH)

            # Vectorization
            train_set, test_set = nb_vectorize(train_set, test_set, features, CATEGORIES.keys())

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
        # context['accuracy'] = '{0:.4f}'.format(accuracy * 100)

        predict_list = test_x.reshape(len(test_x), -1)
        category_list = test_y
        predictions_num = classifier.predict(predict_list)

        context['hr_scores'] = get_scores(category_list, predictions_num, 1)
        context['road_scores'] = get_scores(category_list, predictions_num, 4)
        context['bridge_scores'] = get_scores(category_list, predictions_num, 5)
        context['flood_scores'] = get_scores(category_list, predictions_num, 6)
        context['commend_scores'] = get_scores(category_list, predictions_num, 10)
        context['accuracy'] = (context['hr_scores']['acc'] + context['road_scores']['acc'] + context['bridge_scores']['acc'] + context['flood_scores']['acc'] + context['commend_scores']['acc']) / 5

        for i in range(len(predictions_num)):
            correct = 'Yes' if predictions_num[i] == category_list[i] else 'No'
            context['prediction'].append({
                'id': test_id[i],
                'system_category': CATEGORIES[str(predictions_num[i])],
                'actual_category': CATEGORIES[str(category_list[i])],
                'correct': correct
            })

    return render(request, 'performance.html', context)

def subperformance(request):
    context = {'accuracy': 0.0, 'prediction': [], }

    if request.method == 'POST':
        complaints = load_raw1(RAW_CSV_PATH)

        # Tokenization, Stopword Removal, and Stemming
        i = 1
        for complaint in complaints:
            complaint['body'] = tokenize(complaint['body'])
            #complaint['body'] = ner(complaint['body'])
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
        features = extract_features(train_set, SUBCATEGORIES.keys())
        write_json(features, FEATURES_JSON_PATH)

        # Vectorization
        train_set, test_set = nb_vectorize(train_set, test_set, features, SUBCATEGORIES.keys())

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

        for i in range(len(predictions_num)):
            correct = 'Yes' if predictions_num[i] == category_list[i] else 'No'
            context['prediction'].append({
                'id': test_id[i],
                'system_category': SUBCATEGORIES[str(predictions_num[i])],
                'actual_category': SUBCATEGORIES[str(category_list[i])],
                'correct': correct
            })

    return render(request, 'subperformance.html', context)

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
        features = extract_features(train_set, CATEGORIES.keys())
        write_json(features, FEATURES_JSON_PATH)

        # Vectorization
        train_set, test_set = vectorize(train_set, test_set, features)

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

        for i in range(len(predictions_num)):
            correct = 'Yes' if predictions_num[i] == category_list[i] else 'No'
            context['prediction'].append({
                'id': test_id[i],
                'system_category': CATEGORIES[str(predictions_num[i])],
                'actual_category': CATEGORIES[str(category_list[i])],
                'correct': correct
            })

    return render(request, 'traditional.html', context)

# def generate_report(request):
#     pass

# def categorized(request):
#     return render(request, 'categorized.html')

# def complaints(request):
#     return render(request, 'complaints.html')

# def adminpage(request):
#     return render(request, 'adminpage.html')

def data(request):
    return render(request, 'data.html')