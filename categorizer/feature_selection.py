from collections import Counter

def TFIDF(TF, complaints, term):
	if TF >= 1:
		n = len(complaints)
		x = sum([1 for complaint in complaints if term in complaint['body']])
		return log(TF + 1) * log(n / x)
	else:
		return 0

def DF(vocab, complaints):
	term_DF = dict()
	for term in vocab:
		term_DF[term] = sum([1 for complaint in complaints if term in complaint['body']])

	threshold = 2
	features = [term for term in term_DF.keys() if term_DF[term] > threshold]

	return features

def chi_square(vocab, complaints):
	features = []
	chi_table = dict()
	categories = [
		# 'newsinfo',
		# 'globalnation',
		# 'opinion',
		# 'technology',
		# 'entertainment',
		# 'business',
		# 'sports',
		'1', '4', '5', '6'
	]
	N = len(complaints)
	
	for term in vocab:
		chi_table[term] = dict()
		for category in categories:
			chi_table[term][category] = dict()
			A = 0
			B = 0
			C = 0
			D = 0
			for complaint in complaints:
				if term in complaint['body'] and complaint['category'] == category:
					A += 1
				if term in complaint['body'] and complaint['category'] != category:
					B += 1
				if term not in complaint['body'] and complaint['category'] == category:
					C += 1
				if term not in complaint['body'] and complaint['category'] != category:
					D += 1
			chi_table[term][category]['chi'] = (N * ((A * D) - (C * B))**2) / ((A + C) * (B + D) * (A + B) * (C + D))
			chi_table[term][category]['freq'] = A + C
		chi_table[term]['chi_average'] = float()
		for category in categories:
			P = chi_table[term][category]['freq'] / N
			chi_table[term]['chi_average'] += P * chi_table[term][category]['chi']
		if chi_table[term]['chi_average'] > 5:
			features.append(term)

	print('Extracted {0} features'.format(len(features)))
	return features