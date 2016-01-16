import pickle

from sklearn.cross_validation import cross_val_score
from sklearn.tree import DecisionTreeClassifier

import get_features
import noise


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


def get_union_of_all_but_i(list_of_lists, index):
    res = []
    for j in range(0, len(list_of_lists)):
        if j != index:
            res.extend(list_of_lists[j])
    return res


if __name__ == '__main__':
    x, y = extract_data_from_ads()
    x_temp = x.copy()

    for i in range(0, len(x_temp)):
        x_temp[i].append(y[i])

    noisy_folds, folds = noise.get_noisy_folds(x_temp)

    clf = DecisionTreeClassifier(criterion='entropy', splitter='best', min_samples_split=49)

    for k in range(0, len(noisy_folds)):
        learn_group_x = get_union_of_all_but_i(noisy_folds, k)
        learn_group_y = []
        for l in learn_group_x:
            learn_group_y.append(l.pop())

        curr_tree = clf.fit(learn_group_x, learn_group_y)

        num_of_success = 0
        for m in folds[k]:
            ans = m.pop()
            tree_ans = curr_tree.predict([m])
            m.append(ans)
            if ans == tree_ans:
                num_of_success += 1

        for l in learn_group_x:
            l.append(learn_group_y.pop(0))

        print('fold num: {} | acc: {}'.format(k, num_of_success / (float(len(folds[k])))))
