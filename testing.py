import json
from categorizer.preprocessing import load_raw, tokenize, remove_stopwords, find_closest_word
from nltk.corpus import words, brown, reuters
from nltk import word_tokenize

english = set(brown.words())
english.update(set(reuters.words()))

with open('globals/data/tagalog.json', 'r') as file:
    tagalog = json.load(file)

need_to_translate = set()
no_need_to_translate = set()

complaints = load_raw('globals/data/raw.csv')

i = 1
for complaint in complaints:
    complaint['body'] = tokenize(complaint['body'])
    complaint['body'] = remove_stopwords(complaint['body'])
    
    for word in complaint['body']:
        is_english = word in english
        is_tagalog = word in tagalog
        if not is_english and is_tagalog:
            need_to_translate.add(word)
        elif is_english:
            no_need_to_translate.add(word)
        # else:
        #     print(word + ' : ' + str(find_closest_word(word)))

    print('Finished complaint # ' + str(i))
    i += 1

need_to_translate = list(need_to_translate)

# with open('globals/data/need_to_translate.json', 'w') as file:
#     json.dump(need_to_translate, file)

print('Dumped {0} words.'.format(len(need_to_translate)))
print('Did not dump {0} words.'.format(len(no_need_to_translate)))
