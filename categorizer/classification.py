from sklearn import preprocessing, cross_validation, neighbors


def train_classifier(train_x, train_y):
    clf = neighbors.KNeighborsClassifier()
    clf.fit(train_x, train_y)

    return clf