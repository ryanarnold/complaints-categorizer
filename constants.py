RAW_CSV_PATH = 'globals/data/raw.csv'
RAW_TRAIN_JSON_PATH = 'globals/data/raw_train.json'
RAW_DEVTEST_JSON_PATH = 'globals/data/raw_dev.json'
RAW_EVALTEST_JSON_PATH = 'globals/data/raw_eval.json'
RAW_SUB_TRAIN_JSON_PATH = 'globals/data/raw_sub_train.json'
RAW_SUB_DEVTEST_JSON_PATH = 'globals/data/raw_sub_dev.json'
RAW_SUB_EVALTEST_JSON_PATH = 'globals/data/raw_sub_eval.json'
LOAD_PATH = 'globals/data/'
LOAD_PATHmulti = 'media/data/'
VECTORIZED_TRAIN_CSV_PATH = 'globals/data/vectorized_train.csv'
VECTORIZED_TEST_CSV_PATH = 'globals/data/vectorized_test.csv'
VECTORIZED_SUB_TRAIN_CSV_PATH = 'globals/data/vectorized_sub_train.csv'
VECTORIZED_SUB_TEST_CSV_PATH = 'globals/data/vectorized_sub_test.csv'
VECTORIZED_TRAIN_INPUT_CSV_PATH = 'globals/data/vectorized_train_input.csv'
VECTORIZED_TEST_INPUT_CSV_PATH = 'globals/data/vectorized_test_input.csv'
PREPROCESSED_TRAIN_JSON_PATH = 'globals/data/preprocessed_train.json'
PREPROCESSED_SUB_TEST_JSON_PATH = 'globals/data/preprocessed_test.json'
PREPROCESSED_SUB_TRAIN_JSON_PATH = 'globals/data/preprocessed_sub_train.json'
PREPROCESSED_TEST_JSON_PATH = 'globals/data/preprocessed_sub_test.json'
FEATURES_JSON_PATH = 'globals/data/features.json'
FEATURES_SUB_JSON_PATH = 'globals/data/features_sub.json'

CATEGORIES = {
    '1': 'HR',
    '4': 'ROADS',
    '5': 'BRIDGES',
    '6': 'FLOOD CONTROL',
    '10': 'COMMENDATIONS'
}

SUBCATEGORIES = {
    # '1': 'EMPLOYMENT',
    '2': 'PAYMENT OF SALARIES',
    '3': 'ALLEGATION OF MISBEHAVIOR/MALFEASANCE',
    '5': 'CLAIMS OF BENEFITS',
    '6': 'ALLEGATION OF DEFECTIVE ROAD CONSTRUCTION',
    '7': 'ALLEGATION OF DELAYED ROAD CONSTRUCTION',
    # '8': 'ROAD SAFETY',
    # '9': 'ROAD SIGNS',
    # '10': 'POOR ROAD CONDITION',
    '11': 'REQUEST FOR FUNDING',
    # '13': 'POOR BRIDGE CONDITION',
    # '14': 'BRIDGE SAFETY',
    '15': 'ALLEGATION OF DEFECTIVE BRIDGE CONSTRUCTION',
    '16': 'ALLEGATION OF DELAYED BRIDGE CONSTRUCTION',
    '21': 'CLOGGED DRAINAGE',
    '22': 'DEFECTIVE FLOOD CONTROL CONSTRUCTION',
    # '23': 'FLOOD CONTROL SAFETY',
    '24': 'REQUEST FOR FUNDING',
    '25': 'DELAYED FLOOD CONTROL CONSTRUCTION',
    '26': 'APPLICATION',
    '27': 'REQUEST FOR FUNDING'
}

CATEGORY_CHILDREN = {
    '1': ['2', '3', '5', '26'],
    '4': ['6', '7', '11'],
    '5': ['15', '16', '27'],
    '6': ['21', '22', '24', '25']
}
