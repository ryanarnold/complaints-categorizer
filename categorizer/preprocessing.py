from nltk import word_tokenize, PorterStemmer
from nltk.corpus import stopwords as stopwords_corpus
from nltk.corpus import words, brown, reuters
from nltk.metrics import edit_distance
from random import shuffle
from categorizer.feature_selection import DF, chi_square
import time
import json
from sklearn import preprocessing
import csv
import re

VECTORIZED_CSV_PATH = 'globals/data/vectorized.csv'

punctuations = ['.', ':', ',', ';', '\'', '``', '\'\'', '(', ')', '•', '%',
                '[', ']', '...', '=', '-', '?', '!', '”', '@', '<', '\\\\']
punc = re.compile(r'[^a-zA-Z0-9]')

# Stopwords corpus
stopwords = stopwords_corpus.words('english')
stopwords.append('said')

# Vocabulary of all English words
english = set(brown.words())
english.update(set(reuters.words()))

# Vocabulary of all Tagalog words
with open('globals/data/tagalog.json', 'r') as file:
    tagalog = set(json.load(file))

# Stemmer
porter = PorterStemmer()

# Translation table
with open('globals/data/translated.json') as translated_file:
    trans = json.load(translated_file)

# Road List
with open('globals/data/roadlist.json') as f:
    roadlist = json.load(f)


def hasnum(string):
    for i in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
        if i in string:
            return True
    return False

def load_raw(filepath):
    complaints = []
    with open(filepath, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            complaints.append({
                'id': row[0],
                'body': row[1],
                'category': row[3]
            })

    return complaints

def load_raw1(filepath):
    complaints = []
    with open(filepath, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[4] != 'NULL':
                complaints.append({
                    'id': row[0],
                    'body': row[1],
                    'category': row[4]
                })
            else:
                pass

    return complaints

def load_multi(filepath):
    complaints = []
    with open(filepath, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            complaints.append({
                'id': row[0],
                'body': row[1],
                'category': ''
            })

    return complaints

def tokenize(text):
    tokens = []
    for token in word_tokenize(text):
        token = token.lower().replace('\'', '').replace('“', '').replace('’', '')
        if punc.search(token) == None and token not in punctuations:
            tokens.append(token)
        elif token not in punctuations:
            token = punc.sub(' ', token)
            for t in word_tokenize(token):
                tokens.append(t)

    return tokens

def ner(text):
    i=0
    for i in range(len(text)):
        try:
            if text[i].lower() in roadlist:
                #print ("road na me")
                text[i] = "road"
            else:
                for j in range(len(roadlist)):
                    if text[i].lower() in roadlist[j]:
                        if text[i+1].lower() in roadlist[j]:
                            #print ("YEy road na sya")
                            text[i] = "road"
                        else:
                                #print("Sad")
                            pass
                    else:
                        pass
        except Exception as e:
            #print("Error has occurred", e)
            pass

    return (text)

def remove_stopwords(text):
    text = [token for token in text if token not in stopwords and not hasnum(token)]
    return text

def stem(text):
    stemmed = []
    for token in text:
        if token in english:
            stemmed.append(porter.stem(token))
        elif token in tagalog:
            translation = trans[token].lower() if trans[token] != '-' else token
            for t in word_tokenize(translation):
                if t not in stopwords:
                    stemmed.append(porter.stem(t))
        else:
            # try:
            #     translated = trans[token].lower()
            #     if translated not in stopwords:
            #         stemmed.append(porter.stem(translated.lower()))
            #         # print(translated)
            # except KeyError:
            #     print(token)
            #     stemmed.append(porter.stem(translated.lower()))
            pass
    
    return stemmed

def extract_features(complaints, categories):
    text = list()
    for complaint in complaints:
        text += complaint['body']
    vocab = set(text)
    initial_features = DF(vocab, complaints)    # eliminates too infrequent terms
    informative_features = chi_square(initial_features, complaints, categories)

    return informative_features

def vectorize(train_set, test_set, features):
    vectorized_train_set = []
    for complaint in train_set:
        vector = {}
        for term in features:    
            # TF = complaint['body'].count(term)
            # vector[term] = TFIDF(TF, complaints, term)
            vector[term] = complaint['body'].count(term)
        vectorized_train_set.append({'vector': vector, 'category': complaint['category'], 'id': complaint['id']})

    vectorized_test_set = []
    for complaint in test_set:
        vector = {}
        for term in features:    
            # TF = complaint['body'].count(term)
            # vector[term] = TFIDF(TF, complaints, term)
            vector[term] = complaint['body'].count(term)
        vectorized_test_set.append({'vector': vector, 'category': complaint['category'], 'id': complaint['id']})

    return vectorized_train_set, vectorized_test_set

def nb_vectorize(train_set, test_set, features, categories):
    entire_text = []
    for complaint in train_set:
        for token in complaint['body']:
            entire_text.append(token)

    category_text = {}
    for category in categories:
        category_text[category] = []
        for c in train_set:
            if c['category'] == category:
                for token in c['body']:
                    category_text[category].append(token)

    word_prob = {}
    for word in features:
        if word not in entire_text:
            continue
        word_prob[word] = {}
        for category in categories:
            A = category_text[category].count(word) / len(category_text[category])
            B = len(category_text[category]) / len(entire_text)
            C = entire_text.count(word) / len(entire_text)
            prob = (A * B) / C
            word_prob[word][category] = prob

    vectorized_train_set = list()

    for complaint in train_set:
        # print('Vectorizing ' + complaint['id'])
        vector = {}

        for category in categories:
            prob = float()            
            for word in complaint['body']:
                if word in features:
                    prob += word_prob[word][category]
            n = len(complaint['body'])
            vector[category] = prob / n

        vectorized_train_set.append({'vector': vector, 'category': complaint['category'], 'id': complaint['id']})

    vectorized_test_set = list()

    for complaint in test_set:
        # print('Vectorizing ' + complaint['id'])
        vector = {}

        for category in categories:
            prob = float()            
            for word in complaint['body']:
                if word in word_prob.keys():
                    prob += word_prob[word][category]
            n = len(complaint['body'])
            vector[category] = prob / n

        vectorized_test_set.append({'vector': vector, 'category': complaint['category'], 'id': complaint['id']})

    return vectorized_train_set, vectorized_test_set

def write_csv(complaints, filepath):
    with open(filepath, 'w') as file:
        file.write('id,')
        for category in complaints[0]['vector'].keys():
            file.write(category + ',')
        file.write('category\n')
        for complaint in complaints:
            file.write(complaint['id'] + ',')
            for category in complaints[0]['vector'].keys():
                file.write(str(complaint['vector'][category]) + ',')
            file.write(complaint['category'] + '\n')

def write_json(object, filepath):
    with open(filepath, 'w') as file:
        json.dump(object, file)

def load_json(filepath):
    with open(filepath, 'r') as file:
        obj = json.load(file)
    return obj

def find_closest_word(word):
    best_words = []
    for w in tagalog:
        if w[0] != word[0]:
            continue
        dist = edit_distance(word, w)
        if dist < 3:
            best_words.append(w)
    for w in english:
        if w[0] != word[0]:
            continue
        dist = edit_distance(word, w)
        dist = edit_distance(word, w)
        if dist < 3:
            best_words.append(w)

    return best_words
