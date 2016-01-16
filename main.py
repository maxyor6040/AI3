from sklearn.cross_validation import cross_val_score
from sklearn.tree import DecisionTreeClassifier

import get_features


def current_sample_labeled_ad(list_line):
    return list_line[1558] == "ad.\n"


def extract_data_from_ads():
    file = open('ads/ad.data')

    indices = get_features.get_ads_features(317390805, 317390789)

    x_val = []
    y_val = []

    for line in file:
        list_line = line.split(',')

        sub_list_by_indices = []
        for i in indices:
            sub_list_by_indices.append(list_line[i])
        x_val.append(sub_list_by_indices)

        if current_sample_labeled_ad(list_line):
            y_val.append(1)
        else:
            y_val.append(0)

    return x_val, y_val


if __name__ == '__main__':

    """iris = load_iris()
    cross_val_score(clf, iris.data, iris.target, cv=10)"""
    x, y = extract_data_from_ads()
    clf = DecisionTreeClassifier().fit(x, y)
    print(clf.predict([x[1]]))

