from categorizer import preprocessing
from random import shuffle

complaints = preprocessing.load_raw1('globals/data/complaint.csv')
shuffle(complaints)

half_point = int(len(complaints) * 0.5)
test_point = int(len(complaints) * 0.9)

train_set = complaints[:half_point]
eval_set = complaints[half_point:test_point]
dev_set = complaints[test_point:]

print(len(train_set))
print(len(eval_set))
print(len(dev_set))

preprocessing.write_json(train_set, 'globals/data/raw_sub_train.json')
preprocessing.write_json(eval_set, 'globals/data/raw_sub_eval.json')
preprocessing.write_json(dev_set, 'globals/data/raw_sub_dev.json')
