import copy
import pickle
import random

import math
from sklearn.cross_validation import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import semi_random

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


def get_subset_indices(group, p=0.2):
    return random.sample(range(len(group)), math.floor(0.2 * len(group)))


def committee_predict(current_committee, example):
    fit_count = 0
    for judge in current_committee:
        if judge.predict(example):
            fit_count += 1
    return fit_count > len(current_committee) - fit_count


def calculate_single_tree(noisy_fold_single_tree, folds_single_tree):
    accuracy = 0.0

    clf = DecisionTreeClassifier(criterion='entropy', splitter='best', min_samples_split=49)

    for k in range(0, len(noisy_fold_single_tree)):
        learn_group_x_single_tree = get_union_of_all_but_i(noisy_fold_single_tree, k)
        learn_group_y_single_tree = []
        for l in learn_group_x_single_tree:
            learn_group_y_single_tree.append(l.pop())

        curr_tree_single_tree = clf.fit(learn_group_x_single_tree, learn_group_y_single_tree)

        num_of_success = 0
        for m in folds_single_tree[k]:
            ans = m.pop()
            tree_ans = curr_tree_single_tree.predict([m])
            m.append(ans)
            if ans == tree_ans:
                num_of_success += 1

        for l in learn_group_x_single_tree:
            l.append(learn_group_y_single_tree.pop(0))
        accuracy += num_of_success / (float(len(folds_single_tree[k])))

    accuracy /= float(len(noisy_fold_single_tree))
    print('single tree. acc: {}'.format(k, accuracy))


def calculate_semi_random_committee(noisy_fold_semi, fold_semi, features, committee_size, subset_str):
    accuracy = 0.0
    for k in range(0, len(noisy_fold_semi)):
        learn_group_x = get_union_of_all_but_i(noisy_fold_semi, k)

        committee = []
        for _ in range(0, committee_size):
            committee.append(semi_random.semi_random_id3(features, learn_group_x))

        current_accuracy = 0
        for m in fold_semi[k]:
            if (m[350] == 1) == committee_predict(committee, m):
                current_accuracy += 1
        current_accuracy /= float(len(fold_semi[k]))
        accuracy += current_accuracy
    accuracy /= float(len(noisy_fold_semi))
    print('committee semi-random: subset:{} | size: {} | acc: {}'.format(subset_str, committee_size, accuracy))


def all_semi_random_sub_examples(noisy_fold_semi, fold_semi, features):
    sizes = [1, 3, 5, 7, 9, 11]
    for size in sizes:
        indices = get_subset_indices(noisy_fold_semi)
        noisy_fold_semi_subset = []
        fold_semi_subset = []
        for index in indices:
            noisy_fold_semi_subset.append(noisy_fold_semi[index])
            fold_semi_subset.append(fold_semi[index])

        calculate_semi_random_committee(noisy_fold_semi_subset, fold_semi_subset, features, size, 'examples')


def all_semi_random_sub_features(noisy_fold_semi, fold_semi, features):
    sizes = [1, 3, 5, 7, 9, 11]
    for size in sizes:
        sub_indices = get_subset_indices(features)
        calculate_semi_random_committee(noisy_fold_semi, fold_semi, sub_indices, size, 'features')


if __name__ == '__main__':
    '''this part should be done once'''
    x, y = extract_data_from_ads()
    x_temp = copy.deepcopy(x)
    for i in range(0, len(x_temp)):
        x_temp[i].append(y[i])
    noisy_folds, folds = noise.get_noisy_folds(x_temp)
    '''end of part'''

    '''Arye'''
    all_semi_random_sub_features(noisy_folds, folds, [i for i in range(0, len(x[0]) - 1)])
    '''Max'''
    all_semi_random_sub_examples(noisy_folds, folds, [i for i in range(0, len(x[0]) - 1)])
