from categorizer import preprocessing

test_data = preprocessing.load_json('globals/data/raw_eval.json')

filepath = 'globals/data/raw_eval.csv'
with open(filepath, 'w') as file:
    for c in test_data:
        file.write('"' + c['id'] + '"')
        file.write(',')
        body = c['body']
        body = body.replace('\n', '')
        body = body.replace('"', '')
        body = '"' + body + '"'
        file.write(body)
        file.write('\n')

test_data = preprocessing.load_json('globals/data/raw_sub_eval.json')

filepath = 'globals/data/raw_eval_sub.csv'
with open(filepath, 'w') as file:
    for c in test_data:
        file.write('"' + c['id'] + '"')
        file.write(',')
        body = c['body']
        body = body.replace('\n', '')
        body = body.replace('"', '')
        body = '"' + body + '"'
        file.write(body)
        file.write('\n')
