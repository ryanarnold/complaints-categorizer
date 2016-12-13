from nltk import word_tokenize, PorterStemmer
from nltk.corpus import stopwords as stopwords_corpus
from nltk.corpus import words
from random import shuffle
from categorizer.feature_selection import DF, chi_square
import time
import json
from sklearn import preprocessing, cross_validation

VECTORIZED_CSV_PATH = 'globals/data/vectorized.csv'


def preprocess(complaints):
    punctuations = ['.', ':', ',', ';', '\'', '``', '\'\'', '(', ')',
                    '[', ']', '...', '=', '-', '?', '!']
    stopwords = stopwords_corpus.words('english')
    english = words.words()
    porter = PorterStemmer()
    with open('translated.json') as translated_file:    
            trans = json.load(translated_file)
    for complaint in complaints:
        complaint['body'] = [token.lower().replace('\'', '') for token in word_tokenize(complaint['body']) if token not in punctuations]
        complaint['body'] = [token for token in complaint['body'] if token not in stopwords and not token.isnumeric()]
        complaint['body'] = [token for token in complaint['body'] if token != 'said']
        #complaint['body'] = [porter.stem(token) for token in complaint['body']]
        stemmed = []
        for token in complaint['body']:
                        if token in english:
                                stemmed.append(porter.stem(token))
                        else:
                                #sample = "Kamusta kana"
                                #print(token)
                                try:
                                    translated = trans[token]
                                    stemmed.append(translated)
                                except KeyError:
                                    #print (e)
                                    #print(token)
                                    stemmed.append(token)
        complaint['body'] = stemmed
                
    return complaints

def extract_features(complaints):
    text = list()
    for complaint in complaints:
        text += complaint['body']
    vocab = set(text)
    initial_features = DF(vocab, complaints)    # eliminates too infrequent terms
    informative_features = chi_square(initial_features, complaints)

    return informative_features

def vectorize(complaints, features):
    file = open(VECTORIZED_CSV_PATH, 'w')

    for term in features:
        file.write(term + ',')
    file.write('category\n')

    vectorized_complaints = []
    i = 1
    for complaint in complaints:
        print('Vectorizing complaint # ' + str(i))
        i += 1
        vector = {}
        for term in features:    
            # TF = complaint['body'].count(term)
            # vector[term] = TFIDF(TF, complaints, term)
            vector[term] = complaint['body'].count(term)
        vectorized_complaints.append({'category': complaint['category'], 'vector': vector})
        
        for term in vector.keys():
            file.write(str(vector[term]) + ',')
        file.write(str(complaint['category']))
        file.write('\n')

    file.close()

    return vectorized_complaints

def nb_vectorize(train_set, test_set, features):
    entire_text = []
    for complaint in train_set:
        for token in complaint['body']:
            entire_text.append(token)

    categories = ['1', '4', '5', '6']

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

def execute_preprocessing(complaints):
	# Preprocessing:
    start = time.time()
    print('Preprocessing complaints...')
    preprocessed_complaints = preprocess(complaints)
    print('Finished after {0:.4f} seconds.'.format(time.time() - start))

    # Partition into training set and test set
    shuffle(preprocessed_complaints)
    half_point = int(len(complaints) * 0.8)
    train_set = preprocessed_complaints[:half_point]
    test_set = preprocessed_complaints[half_point:]

    with open('globals/data/preprocessed_train.json', 'w') as file:
        json.dump(train_set, file)
    with open('globals/data/preprocessed_test.json', 'w') as file:
        json.dump(test_set, file)

    # Feature Extraction:
    start = time.time()
    print('Extracting features...')
    features = extract_features(train_set)
    with open('globals/data/features.json', 'w') as features_file:
        json.dump(features, features_file)
    with open('globals/data/features.json', 'r') as features_file:
        features = json.load(features_file)
    print('Finished after {0:.4f} seconds.'.format(time.time() - start))

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

    return vectorized_train_set, vectorized_test_set

def preprocess_single(complaint):
    # Turn it into a list, since the functions only accept lists:
    complaints = []
    complaints.append({
        'id': '',
        'body': complaint,
        'category': ''
    })

    preprocessed = preprocess(complaints)

    with open('globals/data/features.json', 'r') as file:
        features = json.load(file)

    with open('globals/data/preprocessed_train.json', 'r') as file:
        train_set = json.load(file)
    with open('globals/data/preprocessed_test.json', 'r') as file:
        test_set = json.load(file)
    test_set.append(complaints[0])

    vectorized_train_set, vectorized_test_set = nb_vectorize(train_set, test_set, features)

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
