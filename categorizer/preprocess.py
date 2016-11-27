from glob import glob
from nltk import word_tokenize, PorterStemmer, FreqDist, NaiveBayesClassifier
from nltk.classify import accuracy
from nltk.corpus import stopwords as stopwords_corpus
from random import shuffle
from categorizer.feature_selection import TFIDF, DF, chi_square
from math import log

VECTORIZED_CSV_PATH = 'globals/data/vectorized.csv'

def preprocess(complaints):
	punctuations = ['.', ':', ',', ';', '\'', '``', '\'\'', '(', ')', '[', ']', '...', '=', '-', '?', '!']
	stopwords = stopwords_corpus.words('english')
	porter = PorterStemmer()
	for complaint in complaints:
		complaint['body'] = [token.lower().replace('\'', '') for token in word_tokenize(complaint['body']) if token not in punctuations]
		complaint['body'] = [token for token in complaint['body'] if token not in stopwords and not token.isnumeric()]
		complaint['body'] = [token for token in complaint['body'] if token != 'said']
		complaint['body'] = [porter.stem(token) for token in complaint['body']]

	return complaints

def extract_features(complaints):
	text = list()
	for complaint in complaints:
		text += complaint['body']
	vocab = set(text)
	initial_features = DF(vocab, complaints)	# eliminates too infrequent terms
	print('Extracting useful features...')
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

def nb_vectorize(complaints, features):
	entire_text = []
	for complaint in complaints:
		for token in complaint['body']:
			entire_text.append(token)

	categories = ['1', '4', '5', '6']

	category_text = {}
	for category in categories:
		category_text[category] = []
		for c in complaints:
			if c['category'] == category:
				for token in c['body']:
					category_text[category].append(token)

	vectorized_complaints = list()

	for complaint in complaints:
		print('Vectorizing ' + complaint['id'])
		vector = {}

		for category in categories:
			prob = float()			

			for word in complaint['body']:
				A = category_text[category].count(word) / len(category_text[category])
				B = len(category_text[category]) / len(entire_text)
				C = entire_text.count(word) / len(entire_text)
				prob += (A * B) / C

			n = len(complaint['body'])
			vector[category] = prob / n

		vectorized_complaints.append({'vector': vector, 'category': complaint['category'], 'id': complaint['id']})

	return vectorized_complaints

def train(complaints):
	shuffle(complaints)
	half_point = int(len(complaints) / 2)
	train_set = [(complaint['vector'], complaint['category']) for complaint in complaints[:half_point]]
	test_set = [(complaint['vector'], complaint['category']) for complaint in complaints[half_point:]]

	classifier = NaiveBayesClassifier.train(train_set)
	return accuracy(classifier, test_set)