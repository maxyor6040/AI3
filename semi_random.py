import copy
import random
from math import log


class Node:
    """"" left == with feature, right == without feature"""

    def __init__(self, left=None, right=None, feature=-1, result=None):
        self.l = left
        self.r = right
        self.feature = feature
        self.result = result

    """ example is a single vector of features"""

    def predict(self, example):
        if self.result is not None:
            return self.result
        if example[self.feature] == 1:
            return self.l.predict(example)
        else:
            return self.r.predict(example)


def current_example_true(example):
    return example[350] == 1


def positive_examples_count(examples):
    i = 0
    for e in examples:
        if current_example_true(e):
            i += 1
    return i


def count_prob_pos(examples):
    return positive_examples_count(examples) / (float(len(examples)))


def entropy(examples):
    prob_pos = count_prob_pos(examples)
    prob_neg = 1 - prob_pos
    if prob_neg == 0 or prob_pos == 0:
        return 0
    return - (prob_pos * log(prob_pos) + prob_neg * log(prob_neg))


def weighted_choice(choices):
    total = sum(w for c, w in choices.items())
    r = random.uniform(0, total)
    upto = 0
    for c, w in choices.items():
        if upto + w >= r:
            return c
        upto += w
    print("ein kaze mila")
    assert False
    return choices.keys()[0]


def information_gain(feature, examples):
    examples_with_feature = [e for e in examples if e[feature] == '1']
    examples_without_feature = [e for e in examples if e[feature] != '1']
    if len(examples_without_feature) == 0 or len(examples_with_feature) == 0:
        return 0
    weight_with = len(examples_with_feature) / (float(len(examples)))
    weight_without = len(examples_without_feature) / (float(len(examples)))
    return entropy(examples) - weight_with * entropy(examples_with_feature) - weight_without * entropy(
            examples_without_feature)


def calc_res(examples):
    pos_count = positive_examples_count(examples)
    if pos_count > len(examples) - pos_count:
        return Node(result=True)
    return Node(result=False)


def aux(features, examples, m=49, epsilon=0.0000001):
    pos_count = positive_examples_count(examples)
    if len(examples) <= 49:
        return calc_res(examples)

    if pos_count == len(examples):
        return Node(result=True)
    if pos_count == 0:
        return Node(result=False)
    if len(features) == 0:
        return calc_res(examples)

    map_of_weights = {feat: information_gain(feat, examples) + epsilon for feat in features}
    chosen_feature = weighted_choice(map_of_weights)

    left_examples = [e for e in examples if e[chosen_feature] == '1']
    right_examples = [e for e in examples if e[chosen_feature] != '1']
    if len(left_examples) == 0 or right_examples == 0:
        return calc_res(examples)

    features.remove(chosen_feature)
    l = aux(features, left_examples)
    r = aux(features, right_examples)
    res = Node(l, r, chosen_feature)
    features.append(chosen_feature)
    return res


def semi_random_id3(examples):
    return aux([i for i in range(0, len(examples[0]) - 1)], examples)
