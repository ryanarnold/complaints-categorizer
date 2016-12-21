import requests
import json
from microsofttranslator import Translator
from requests.exceptions import ConnectionError

with open('globals/data/need_to_translate.json', 'r') as sample_file:
    words = json.load(sample_file)
    
translated = dict()
translator = Translator('termtranslation_123', '5iFGxhem9bewVDqd4m6mMvT1UAHRueLrR71roc8SRHI=')
for w in words:
    try:
        print (w, end='\t\t')
        translated[w] = translator.translate(w, "en", "fil-PH")
        print (translated[w])
    except (ConnectionError, ValueError) as e:
        print (e)
        input()
        translated[w] = w

with open('globals/data/translated.json', 'w') as file:
    json.dump(translated, file)
