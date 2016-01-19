import copy
import pickle
import random

import math
import threading

from sklearn.cross_validation import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from zmq.auth import thread

import semi_random

import get_features
import noise

sizes = [1, 3, 5, 7, 9, 11]


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
    return random.sample(range(len(group)), math.ceil(p * len(group)))


def get_sub_group_of_examples(all_examples):
    '''
    indices = get_subset_indices(noisy_folds_full[0])
    noisy_fold_semi_subset = []
    fold_semi_subset = []
    for _ in range(0, len(noisy_folds_full)):
        noisy_fold_semi_subset.append([])
        fold_semi_subset.append([])

    for noisy_vec_num in range(0, len(noisy_folds_full)):
        for index in indices:
            noisy_fold_semi_subset[noisy_vec_num].append(noisy_folds_full[noisy_vec_num][index])
            fold_semi_subset[noisy_vec_num].append(folds_full[noisy_vec_num][index])

    return noisy_fold_semi_subset, fold_semi_subset
    '''

    indices = get_subset_indices(all_examples)
    subset = []
    for curr_i in indices:
        subset.append(all_examples[curr_i])

    return subset


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


def calculate_semi_random_committee(noisy_fold_semi, fold_semi, features, committee_size, is_subset_of_examples):
    accuracy = 0.0
    for k in range(0, len(noisy_fold_semi)):
        learn_group_x = get_union_of_all_but_i(noisy_fold_semi, k)

        committee = []
        for _ in range(0, committee_size):
            curr_examples = learn_group_x
            curr_feats = features
            if is_subset_of_examples:
                curr_examples = get_sub_group_of_examples(learn_group_x)
            else:
                curr_feats = get_subset_indices(features)

            committee.append(semi_random.semi_random_id3(curr_feats, curr_examples))

        current_accuracy = 0
        for m in fold_semi[k]:
            if (m[350] == 1) == committee_predict(committee, m):
                current_accuracy += 1
        current_accuracy /= float(len(fold_semi[k]))
        accuracy += current_accuracy
    accuracy /= float(len(noisy_fold_semi))
    print('committee semi-random: subset:{} | size: {} | acc: {}'.format(
            'examples' if is_subset_of_examples else 'features', committee_size, accuracy))


def all_semi_random_sub_examples(noisy_fold_semi, fold_semi, features):
    for size in sizes:
        calculate_semi_random_committee(noisy_fold_semi, fold_semi, features, size, True)


def all_semi_random_sub_features(noisy_fold_semi, fold_semi, features):
    for size in sizes:
        calculate_semi_random_committee(noisy_fold_semi, fold_semi, features, size, False)


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
