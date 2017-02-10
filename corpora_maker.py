from nltk.corpus import stopwords as stopwords_corpus
from nltk.corpus import words, brown, reuters
from categorizer.preprocessing import write_json

# Stopwords corpus
stopwords = stopwords_corpus.words('english')
stopwords.append('said')
stopwords.append('po')
stopwords.append('pa')
stopwords.append('sa')
stopwords.append('may')
stopwords.append('para')

print(len(stopwords))
write_json(stopwords, 'globals/data/stopwords.json')

# Vocabulary of all English words
english = set(brown.words())
english.update(set(reuters.words()))
english.add('overpass')

english = list(english)
for i in range(len(english)):
    english[i] = english[i].lower()

print(len(english))
write_json(english, 'globals/data/english.json')
