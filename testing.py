import json
from pprint import pprint

with open('translated.json') as translated_file:    
    trans = json.load(translated_file)

token = "anoba"
try:
    translated = trans[token]
    print (translated)
except KeyError:
    print (token)
