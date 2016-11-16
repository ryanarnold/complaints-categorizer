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
	# print('Extracting useful features...')
	# informative_features = chi_square(initial_features, complaints)

	return initial_features

def vectorize(complaints, features):
	file = open(VECTORIZED_CSV_PATH, 'w')
	class_values = {
		'newsinfo': 1,
		'globalnation': 2,
		'opinion': 3,
		'technology': 4,
		'entertainment': 5,
		'business': 6,
		'sports': 7,

	}

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

def train(complaints):
	shuffle(complaints)
	half_point = int(len(complaints) / 2)
	train_set = [(complaint['vector'], complaint['category']) for complaint in complaints[:half_point]]
	test_set = [(complaint['vector'], complaint['category']) for complaint in complaints[half_point:]]

	classifier = NaiveBayesClassifier.train(train_set)
	return accuracy(classifier, test_set)