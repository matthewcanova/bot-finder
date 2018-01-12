import math


########################
# Entropy and Info Gain
########################

def split_data(examples, feature):
    yes = []
    no = []
    if feature == 0:
        for example in examples:
            if example[0] == 1:
                yes.append(example)
            else:
                no.append(example)
    else:
        for example in examples:
            if feature in example:
                yes.append(example)
            else:
                no.append(example)
    return yes, no


def entropy(examples):
    entropy_total = 0
    total = len(examples)
    items = [0, 0]
    for example in range(len(examples)):
        if examples[example][0] == 1:
            items[1] += 1
        if examples[example][0] == -1:
            items[0] += 1
    if items[0] != 0 and items[1] != 0:
        for item in items:
            delta = ((item/total) * -1) * math.log2(item/total)
            entropy_total += delta
    return entropy_total


def majority_error(examples):
    error_total = 0
    total = len(examples)
    items = [0, 0]
    for example in range(len(examples)):
        if examples[example][0] == 1:
            items[1] += 1
        if examples[example][0] == -1:
            items[0] += 1
    if items[0] != 0 and items[1] != 0:
        if items[0] > items[1]:
            error_total = 1 - (items[0]/total)
        else:
            error_total = 1 - (items[1]/total)
    return error_total


def info_gain(examples, feature, ent="entropy"):

    entropy_total = 0

    total = len(examples)

    yes, no = split_data(examples, feature)

    if ent == "entropy":
        entropy_total += (len(yes) / total) * entropy(yes)
        entropy_total += (len(no) / total) * entropy(no)
    else:
        entropy_total += (len(yes) / total) * majority_error(yes)
        entropy_total += (len(no) / total) * majority_error(no)

    return entropy_total


def info_gain_best(examples, features, ent="entropy"):
    # Entropy of whole set
    if ent == "entropy":
        item_entropy = entropy(examples)
    else:
        item_entropy = majority_error(examples)

    # for each attribute, store a gain value
    gain_results = {}

    for feature in features:
            gain_results[feature] = item_entropy - info_gain(examples, feature, ent)

    keys = list(gain_results.keys())
    values = list(gain_results.values())

    return keys[values.index(max(values))]


#################
# ID3 Algorithm
#################


def id3(examples, features, depth):

    # calculate the data split by label
    yes, no = split_data(examples, 0)

    common_label = 1 if len(yes) > len(no) else -1

    # If there is only one label, return a node with that
    if len(yes) == 0:
        return {"label": common_label}
    if len(no) == 0:
        return {"label": common_label}

    # If we are out of features or depth, return node with majority label
    if len(features) < 1:
        return {"label": common_label}
    if depth < 1:
        return {"label": common_label}

    # calculate all remaining attribute gains
    best_gain = info_gain_best(examples, features)

    node = {"label": best_gain}

    # calculate the data split by attribute label
    attr_yes, attr_no = split_data(examples, best_gain)

    features.remove(best_gain)

    if attr_yes:
        node[1] = id3(attr_yes, features, depth-1)
    else:
        node[1] = {"label": common_label}

    if attr_no:
        node[0] = id3(attr_no, features, depth-1)
    else:
        node[0] = {"label": common_label}

    return node


############
# INTERFACE
############

def train_tree(examples, num_features, depth=999):

    features = set()

    for x in range(1, num_features+1):
        features.add(x)

    return id3(examples, features, depth)


def run_tree(tree, example):
    curr_label = tree["label"]
    node = tree
    while 0 in node:
        if curr_label in example:
            branch = 1
        else:
            branch = 0
        node = node[branch]
        curr_label = node["label"]
    return curr_label


def test_tree(tree, examples):
    success = 0
    for example in examples:
        result = run_tree(tree, example)
        if (example[0] * result) > 0:
            success += 1

    return success / len(examples)
