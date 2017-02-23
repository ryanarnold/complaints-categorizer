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
from constants import *

do_preprocessing = False


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
        subcategory = ''

        if category != 'COMMENDATIONS':
            preprocess_subcategory(str(predictions_num[-1]), additionals=[complaintmessage])
            # Get the vectorized data, to prepare it for classification:
            train_x = get_x(VECTORIZED_SUB_TRAIN_CSV_PATH)
            train_y = get_y(VECTORIZED_SUB_TRAIN_CSV_PATH)
            test_x = get_x(VECTORIZED_SUB_TEST_CSV_PATH)
            test_y = get_y(VECTORIZED_SUB_TEST_CSV_PATH)
            test_id = get_id(VECTORIZED_SUB_TEST_CSV_PATH)
            classifier = train_classifier(train_x, train_y)

            predict_list = test_x.reshape(len(test_x), -1)
            predictions_num = classifier.predict(predict_list)
            subcategory = SUBCATEGORIES[str(predictions_num[-1])]
    else:
        complaint = ''
        category = ''
        subcategory = ''
        
    return render(request, 'categorizer.html', {'complaint': complaintmessage, 
        'category': category, 'subcategory': subcategory})

def multicategorizer(request):
    context = {
        'accuracy': 0.0, 
        'prediction': []
    }
    complaints = []
    complaintpath = ""
    saved = False
    if request.method == 'POST':
        complaint = request.FILES['csvfile'].name
        complaintpath = LOAD_PATH + complaint
        print (complaint)
        print (LOAD_PATH)
        print(complaint)
        print(complaintpath)
        inputcomplaints = load_multi(complaintpath)

        # Tokenization, Stopword Removal, and Stemming
        train_set = load_json(RAW_TRAIN_JSON_PATH)
        train_set = preprocess_bulk(train_set)
        print(inputcomplaints)
        test_set = preprocess_bulk(inputcomplaints)

        # Feature extraction (needed in vectorization)
        # features = extract_features(train_set, CATEGORIES.keys())
        # write_json(features, FEATURES_JSON_PATH)
        features = load_json(FEATURES_JSON_PATH)

        # Vectorization
        train_set, test_set = nb_vectorize(train_set, test_set, features, CATEGORIES.keys())

        # Put vectorized data in csv (sklearn reads from csv kasi)
        write_csv(train_set, VECTORIZED_TRAIN_INPUT_CSV_PATH)
        write_csv(test_set, VECTORIZED_TEST_INPUT_CSV_PATH)

        # Get the vectorized data, to prepare it for classification:
        df = pd.read_csv(VECTORIZED_TRAIN_INPUT_CSV_PATH)

        X_train = np.array(df.drop(['category', 'id'], 1))
        y_train = np.array(df['category'])

        df = pd.read_csv(VECTORIZED_TEST_INPUT_CSV_PATH)

        X_test = np.array(df.drop(['category', 'id'], 1))
        y_test = np.array(df['category'])
        test_id = get_id(VECTORIZED_TEST_INPUT_CSV_PATH)
        classifier = train_classifier(X_train, y_train)


        # Prepare output for template:

        predict_list = X_test.reshape(len(X_test), -1)
        predictions_num = classifier.predict(predict_list)
        predictions_subnum = []

        # FOR SUBCATEGORY
        inputcomplaints = load_multi(complaintpath)
        for i in range(len(predictions_num)):
            if str(predictions_num[i]) == '10':
                predictions_subnum.append(99)
                continue
            preprocess_subcategory(str(predictions_num[i]), additionals=[inputcomplaints[i]['body']])
            train_x = get_x(VECTORIZED_SUB_TRAIN_CSV_PATH)
            train_y = get_y(VECTORIZED_SUB_TRAIN_CSV_PATH)
            test_x = get_x(VECTORIZED_SUB_TEST_CSV_PATH)
            test_y = get_y(VECTORIZED_SUB_TEST_CSV_PATH)
            test_id = get_id(VECTORIZED_SUB_TEST_CSV_PATH)
            classifier = train_classifier(train_x, train_y)

            predict_list = test_x.reshape(len(test_x), -1)
            predicts = classifier.predict(predict_list)
            predictions_subnum.append(predicts[-1])

        for i in range(len(predictions_num)):
            context['prediction'].append({
                'id': test_id[i],
                'body': inputcomplaints[i]['body'],
                'system_category': CATEGORIES[str(predictions_num[i])],
                'system_subcategory': SUBCATEGORIES[str(predictions_subnum[i])] if predictions_subnum[i] != 99 else ''
            })

        filepath = 'globals/static/report.csv'
        with open(filepath, 'w') as file:
            for c in context['prediction']:
                file.write(c['id'])
                file.write(',')
                file.write('"' + c['body'] + '"')
                file.write(',')
                file.write('"' + c['system_category'] + '"')
                file.write(',')
                file.write('"' + c['system_subcategory'] + '"')
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
            train_set = load_json(RAW_TRAIN_JSON_PATH)
            test_set = load_json(RAW_EVALTEST_JSON_PATH)

            train_set = preprocess_bulk(train_set)
            test_set = preprocess_bulk(test_set)

            # Partition into training set and test set
            # shuffle(complaints)
            # half_point = int(len(complaints) * 0.8)
            # train_set = complaints[:half_point]
            # test_set = complaints[half_point:]
            write_json(train_set, PREPROCESSED_TRAIN_JSON_PATH)
            write_json(test_set, PREPROCESSED_TEST_JSON_PATH)

            # Feature extraction (needed in vectorization)
            features = extract_features(train_set, CATEGORIES.keys())
            write_json(features, FEATURES_JSON_PATH)
            # features = load_json(FEATURES_JSON_PATH)

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
        context['accuracy'] = '{0:.4f}'.format(accuracy * 100)

        predict_list = test_x.reshape(len(test_x), -1)
        category_list = test_y
        predictions_num = classifier.predict(predict_list)

        context['hr_scores'] = get_scores(category_list, predictions_num, 1)
        context['road_scores'] = get_scores(category_list, predictions_num, 4)
        context['bridge_scores'] = get_scores(category_list, predictions_num, 5)
        context['flood_scores'] = get_scores(category_list, predictions_num, 6)
        context['commend_scores'] = get_scores(category_list, predictions_num, 10)
        # context['accuracy'] = (context['hr_scores']['acc'] + context['road_scores']['acc'] + context['bridge_scores']['acc'] + context['flood_scores']['acc'] + context['commend_scores']['acc']) / 5

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
    context = {'accuracy': {'1': 0.0, '4': 0.0, '5': 0.0, '6': 0.0}, 'prediction': [], 'ave_acc': 0.0}
    context['categories'] = CATEGORIES
    context['subcategories'] = SUBCATEGORIES
    context['category_children'] = CATEGORY_CHILDREN
    context['scores'] = {}
    for category in CATEGORY_CHILDREN.keys():
        for s in CATEGORY_CHILDREN[category]:
            context['scores'][s] = {}
            context['scores'][s] = {'TP': 0.0, 'TN': 0.0, 'FP': 0.0, 'FN': 0.0, 'p': 0.0,'r': 0.0, 'F': 0.0, 'acc': 0.0}

    if request.method == 'POST':
        classifiers = {}
        bad_complaints = []
        for category in ['1', '4', '5', '6']:
            if category == '10':
                continue
            preprocess_subcategory(category)

            # Get the vectorized data, to prepare it for classification:
            train_x = get_x(VECTORIZED_SUB_TRAIN_CSV_PATH)
            train_y = get_y(VECTORIZED_SUB_TRAIN_CSV_PATH)
            test_x = get_x(VECTORIZED_SUB_TEST_CSV_PATH)
            test_y = get_y(VECTORIZED_SUB_TEST_CSV_PATH)
            test_id = get_id(VECTORIZED_SUB_TEST_CSV_PATH)
            classifiers[category] = train_classifier(train_x, train_y)
            context['accuracy'][category] = '{0:0.4f}'.format(classifiers[category].score(test_x, test_y))
            context['ave_acc'] += classifiers[category].score(test_x, test_y)

            predict_list = test_x.reshape(len(test_x), -1)
            category_list = test_y
            predictions_num = classifiers[category].predict(predict_list)

            for s in CATEGORY_CHILDREN[category]:
                context['scores'][s] = get_scores(category_list, predictions_num, int(s))

            raw_test_set = [
                c for c in load_json(RAW_SUB_EVALTEST_JSON_PATH)
                if c['category'] in CATEGORY_CHILDREN[category]
            ]
            test_set = preprocess_bulk(list(raw_test_set))
            raw_test_set = [
                c for c in load_json(RAW_SUB_EVALTEST_JSON_PATH)
                if c['category'] in CATEGORY_CHILDREN[category]
            ]
            for i in range(len(predictions_num)):
                if predictions_num[i] != category_list[i]:
                    bad_complaint = {}
                    bad_complaint['id'] = test_id[i]
                    bad_complaint['actual'] = SUBCATEGORIES[str(category_list[i])]
                    bad_complaint['predicted'] = SUBCATEGORIES[str(predictions_num[i])]
                    bad_complaint['body'] = find_complaint(test_id[i], raw_test_set)['body']
                    bad_complaint['tokens'] = test_set[i]['body']
                    # bad_complaint['vector'] = list(predict_list[i])
                    bad_complaints.append(bad_complaint)

            for i in range(len(predictions_num)):
                correct = 'Yes' if predictions_num[i] == category_list[i] else 'No'
                context['prediction'].append({
                    'id': test_id[i],
                    'system_category': SUBCATEGORIES[str(predictions_num[i])],
                    'actual_category': SUBCATEGORIES[str(category_list[i])],
                    'correct': correct
                })

            # for f in features:
            #     print(f, end=', ')
            # print(category)
            # input()

            # for i in range(len(predictions_num)):
            #     correct = 'Yes' if predictions_num[i] == category_list[i] else 'No'
            #     context['prediction'].append({
            #         'id': test_id[i],
            #         'system_category': SUBCATEGORIES[str(predictions_num[i])],
            #         'actual_category': SUBCATEGORIES[str(category_list[i])],
            #         'correct': correct
            #     })
        write_json(bad_complaints, 'bad_complaints.json')
        context['ave_acc'] /= 4
        context['ave_acc'] = '{0:.4f}'.format(context['ave_acc'])

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
