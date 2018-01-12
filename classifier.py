import random
import math

import decision_tree


###################
# HELPER FUNCTIONS
###################

def count_features(data):
    max_features = 0
    for example in data:
        max_example = max(example)
        if max_example > max_features:
            max_features = max_example
    return max_features


def process_data(data):
    data = data.splitlines()

    examples = []

    for line in data:
        line = line.split()
        example = {}
        for x in range(len(line)):
            if x == 0:
                if line[0] == "0":
                    example[x] = -1
                else:
                    example[x] = 1
            else:
                entry = line[x].split(':')
                target = int(entry[0])
                value = float(entry[1])
                example[target] = value
        examples.append(example)
    return examples


def dot_prod(vec1, vec2):
    sum = 0
    for key, value in vec1.items():
        if key in vec2:
            sum += (vec2[key] * value)
    return sum


def count_label(examples, feature, label):
    count = 0
    if label == "pos":
        if feature == 0:
            for example in examples:
                if example[feature] == 1:
                    count += 1
        else:
            for example in examples:
                if example[0] == 1 and feature in example:
                    count += 1
    elif label == "neg":
        if feature == 0:
            for example in examples:
                if example[feature] == -1:
                    count += 1
        else:
            for example in examples:
                if example[0] == -1 and feature in example:
                    count += 1

    return count


def dict_max_val(dictionary):
    vals = list(dictionary.values())
    keys = list(dictionary.keys())
    return keys[vals.index(max(vals))]


######################
# CLASSIFIER TRAINERS
######################

def simple_perceptron(examples, features, rate, epochs, total=True, updates=False):

    w = {}

    epoch_weights = []
    epoch_biases = []

    for x in range(features):
        w[x] = random.uniform(-.01, .01)

    b = random.uniform(-.01, .01)

    num_updates = 0

    for x in range(epochs):
        random.shuffle(examples)
        for example in examples:
            label = example[0]
            example.pop(0, None)

            # predict the examples outcome with the weight vector
            outcome = dot_prod(example, w) + b

            if outcome * label <= 0:
                #update = [x * rate * label for key, x in example]
                for key, val in example:
                    w[key] += val * rate * label
                b = b + (rate * label)
                num_updates += 1
        epoch_weights.append(w)
        epoch_biases.append(b)

    if updates:
        print(num_updates/epochs)

    if total:
        return w, b
    else:
        return epoch_weights, epoch_biases


def dynamic_perceptron(examples, features, rate, epochs, total=True, updates=False):

    w = {}

    epoch_weights = []
    epoch_biases = []

    for x in range(1, features+1):
        w[x] = random.uniform(-.01, .01)

    b = random.uniform(-.01, .01)

    count = 0
    num_updates = 0

    for x in range(epochs):
        random.shuffle(examples)
        for example in examples:

            dyn_rate = rate / (1 + count)
            count += 1

            label = example[0]
            example.pop(0, None)

            # predict the examples outcome with the weight vector
            outcome = dot_prod(example, w) + b

            if outcome * label <= 0:
                for key, value in example.items():
                    w[key] += value * dyn_rate * label
                b = b + (dyn_rate * label)
                num_updates += 1
            example[0] = label
        epoch_weights.append(w)
        epoch_biases.append(b)

    if updates:
        print(num_updates/epochs)

    if total:
        return w, b
    else:
        return epoch_weights, epoch_biases


def margin_perceptron(examples, features, rate, epochs, margin, total=True, updates=False):

    w = []

    epoch_weights = []
    epoch_biases = []

    for x in range(features):
        w.append(random.uniform(-.01, .01))

    b = random.uniform(-.01, .01)

    count = 0
    num_updates = 0

    for x in range(epochs):
        random.shuffle(examples)
        for example in examples:

            dyn_rate = rate / (1 + count)
            count += 1

            features = example[1:]
            label = example[0]

            # predict the examples outcome with the weight vector
            outcome = dot_prod(features, w) + b

            if outcome * label <= margin:
                update = [x * dyn_rate * label for x in features]
                w = [sum(x) for x in zip(w, update)]
                b = b + (dyn_rate * label)
                num_updates += 1
        epoch_weights.append(w)
        epoch_biases.append(b)

    if updates:
        print(num_updates/epochs)

    if total:
        return w, b
    else:
        return epoch_weights, epoch_biases


def averaged_perceptron(examples, features, rate, epochs, total=True, updates=False):

    w = []

    epoch_weights = []
    epoch_biases = []

    for x in range(features):
        w.append(random.uniform(-.01, .01))

    b = random.uniform(-.01, .01)

    w_a = [0] * features
    b_a = 0

    num_updates = 0

    for x in range(epochs):
        random.shuffle(examples)
        for example in examples:
            features = example[1:]
            label = example[0]

            # predict the examples outcome with the weight vector
            outcome = dot_prod(features, w) + b

            if outcome * label <= 0:
                update = [x * rate * label for x in features]
                w = [sum(x) for x in zip(w, update)]
                b = b + (rate * label)
                num_updates += 1

            w_a = [sum(x) for x in zip(w, w_a)]
            b_a = b_a + b
        examples_num = len(examples) * (x+1)
        epoch_weights.append([x/examples_num for x in w_a])
        epoch_biases.append(b_a/examples_num)

    if updates:
        print(num_updates/epochs)

    if total:
        return [x/(len(examples)*epochs) for x in w_a], b_a/(len(examples)*epochs)
    else:
        return epoch_weights, epoch_biases


def naive_bayes(examples, num_features):
    w = {}

    pos_count = count_label(examples, 0, "pos")
    neg_count = count_label(examples, 0, "neg")

    for x in range(1, num_features + 1):
        pos = count_label(examples, x, "pos")
        neg = count_label(examples, x, "neg")
        w[x] = (pos, neg)

    b = (pos_count, neg_count)

    return w, b


def bagged_forests(examples, num_features, num_trees=1000, depth=999):
    decision_trees = []

    count = 1
    for repeat_num_trees in range(num_trees):
        print("Training Tree " + str(count) + " out of " + str(num_trees) + " trees")
        count += 1
        random_samples = []
        for repeat_num_samples in range(100):
            random_samples.append(random.choice(examples))
        tree = decision_tree.train_tree(random_samples, num_features, depth=depth)
        decision_trees.append(tree)

    return decision_trees


def support_vector_machine(examples, num_features, rate, epochs, smoother):

    w = {}

    for x in range(1, num_features+1):
        w[x] = 0

    b = 0

    count = 0

    for x in range(epochs):
        print("SVM training epoch " + str(x) + "\n")

        rate = rate / (1 + count)
        count += 1

        random.shuffle(examples)

        for example in examples:

            label = example[0]

            example.pop(0, None)

            outcome = label * (dot_prod(example, w) + b)

            if outcome <= 1:
                for key, value in w.items():
                    if key in example:
                        w[key] = ((1 - rate) * w[key]) + (rate * smoother * label)
                    else:
                        w[key] = ((1 - rate) * w[key])
                b = ((1 - rate) * b) + (rate * smoother * label)
            else:
                for key, value in w.items():
                    w[key] = ((1 - rate) * w[key])
                b = ((1 - rate) * b)

            example[0] = label

    return w, b


def logistic_regression(examples, num_features, rate, epochs, smoother):

    w = {}

    for x in range(1, num_features + 1):
        w[x] = 0

    b = 0

    count = 0

    for x in range(epochs):
        print("Logistic Regression training epoch " + str(x) + "\n")

        rate = rate / (1 + count)
        count += 1

        random.shuffle(examples)

        for example in examples:

            y_i = example[0]

            for feature, weight in w.items():
                x_i = 1 if feature in example else 0
                # print("Label: " + str(y_i))
                # print("Example: " + str(x_i))
                # print("Weight: " + str(weight))
                # print("Rate: " + str(rate))
                top_gradient = rate * y_i * x_i * math.exp((-1 * y_i) * weight * x_i)
                bot_gradient = 1 + math.exp((-1 * y_i) * weight * x_i)
                w[feature] = (2 / smoother) * weight + (top_gradient / bot_gradient)

            top_gradient = rate * y_i * math.exp((-1 * y_i) * b)
            bot_gradient = 1 + math.exp((-1 * y_i) * b)
            b = (2 / smoother) * b + (top_gradient / bot_gradient)

    return w, b


######################
# CLASSIFIER DECIDERS
######################

def run_perceptron(features, weights, bias):

    result = dot_prod(features, weights) + bias

    if result > 0:
        return 1
    else:
        return 0


def decide_bayes(example, weights, bias, smoother=0.1):

    pos_prior = math.log(bias[0] / (bias[0]+bias[1]))
    neg_prior = math.log(bias[1] / (bias[0]+bias[1]))

    for feature in example:
        if feature != 0:
            pos_prior += math.log((weights[feature][0] + smoother) / (bias[0] + (2 * smoother)))
            neg_prior += math.log((weights[feature][1] + smoother) / (bias[1] + (2 * smoother)))

    if pos_prior > neg_prior:
        return 1
    else:
        return 0


def decide_forest(example, decision_trees):
    yes = 0
    no = 0
    for tree in decision_trees:
        result = decision_tree.run_tree(tree, example)
        if result == 1:
            yes += 1
        else:
            no += 1

    return 1 if yes > no else 0


###################
# CROSS-VALIDATION
###################

def cross_validate(training_data, mode="log"):

    features = count_features(training_data)

    step = len(training_data) // 5

    data = ([training_data[0:step],
             training_data[step:2*step],
             training_data[2*step:3*step],
             training_data[3*step:4*step],
             training_data[4*step:len(training_data)],
             ])

    if mode == "log":
        learning_rates = {1: 0.0, 0.1: 0.0, 0.01: 0.0, 0.001: 0.0, 0.0001: 0.0, 0.00001: 0.0}
        # smoothers = {0.1: 0.0, 1: 0.0, 10: 0.0, 100: 0.0, 1000: 0.0, 10000: 0.0}
        smoothers = {10: 0.0, 100: 0.0, 1000: 0.0, 10000: 0.0}

        curr_best_rate = 0.1
        curr_best_smoother = 10

        print("Starting " + mode + " Cross-Validation with Best Rate = " + str(curr_best_rate) +
              " and Best Smoother = " + str(curr_best_smoother) + '\n')
        for rate in learning_rates:
            succ_rates = []
            for i in range(0, 5):
                test_data = data[i]
                training_data = []
                for j in range(0, 5):
                    if j is not i:
                        training_data += data[j]

                weights, bias = logistic_regression(training_data, features, rate=rate, epochs=5, smoother=curr_best_smoother)

                succ_rates.append(test_classifier(test_data, weights, bias, mode))

            learning_rates[rate] = sum(succ_rates) / len(succ_rates)

        curr_best_rate = dict_max_val(learning_rates)

        for smoother in smoothers:
            succ_rates = []
            for i in range(0, 5):
                test_data = data[i]
                training_data = []
                for j in range(0, 5):
                    if j is not i:
                        training_data += data[j]

                weights, bias = logistic_regression(training_data, features, rate=curr_best_rate, epochs=5, smoother=smoother)

                succ_rates.append(test_classifier(test_data, weights, bias, mode))

            smoothers[smoother] = sum(succ_rates) / len(succ_rates)

        curr_best_smoother = dict_max_val(smoothers)

        with open('cross-validate-' + mode + '.trace', 'w') as file:
            for rate in learning_rates:
                file.write("Rate: " + str(rate) + "\nAccuracy: " + str(learning_rates[rate]) + '\n')
            for smooth in smoothers:
                file.write("Smoother: " + str(smooth) + "\nAccuracy: " + str(smoothers[smooth]) + '\n')

    if mode == "svm":
        learning_rates = {0.1: 0.0, 0.01: 0.0, 0.001: 0.0, 0.0001: 0.0, 0.00001: 0.0}
        # smoothers = {0.1: 0.0, 1: 0.0, 10: 0.0, 100: 0.0, 1000: 0.0, 10000: 0.0}
        smoothers = {0.1: 0.0, 0.01: 0.0, 0.001: 0.0, 0.0001: 0.0}

        curr_best_rate = 0.1
        curr_best_smoother = 0.0001

        print("Starting " + mode + " Cross-Validation with Best Rate = " + str(curr_best_rate) +
              " and Best Smoother = " + str(curr_best_smoother) + '\n')
        for rate in learning_rates:
            succ_rates = []
            for i in range(0, 5):
                test_data = data[i]
                training_data = []
                for j in range(0, 5):
                    if j is not i:
                        training_data += data[j]

                weights, bias = support_vector_machine(training_data, features, rate=rate, epochs=5, smoother=curr_best_smoother)

                succ_rates.append(test_classifier(test_data, weights, bias, mode))

            learning_rates[rate] = sum(succ_rates) / len(succ_rates)

        curr_best_rate = dict_max_val(learning_rates)

        for smoother in smoothers:
            succ_rates = []
            for i in range(0, 5):
                test_data = data[i]
                training_data = []
                for j in range(0, 5):
                    if j is not i:
                        training_data += data[j]

                weights, bias = support_vector_machine(training_data, features, rate=curr_best_rate, epochs=5, smoother=smoother)

                succ_rates.append(test_classifier(test_data, weights, bias, mode))

            smoothers[smoother] = sum(succ_rates) / len(succ_rates)

        curr_best_smoother = dict_max_val(smoothers)

        with open('cross-validate-' + mode + '.trace', 'w') as file:
            for rate in learning_rates:
                file.write("Rate: " + str(rate) + "\nAccuracy: " + str(learning_rates[rate]) + '\n')
            for smooth in smoothers:
                file.write("Smoother: " + str(smooth) + "\nAccuracy: " + str(smoothers[smooth]) + '\n')


############
# INTERFACE
############

def train_classifier(training_data, mode="bayes", learn_rate=0.1, epochs=20):

    features = count_features(training_data)
    bias = 0

    # Train each mode over epochs using training data
    if mode == "simple":
        weights, bias = simple_perceptron(training_data, features, learn_rate, epochs)
    if mode == "dynamic":
        weights, bias = dynamic_perceptron(training_data, features, learn_rate, epochs)
    if mode == "margin":
        weights, bias = margin_perceptron(training_data, features, learn_rate, epochs, 0.1)
    if mode == "average":
        weights, bias = averaged_perceptron(training_data, features, learn_rate, epochs)
    if mode == "bayes":
        weights, bias = naive_bayes(training_data, features)
    if mode == "forest":
        weights = bagged_forests(training_data, features)
    if mode == "log":
        weights, bias = logistic_regression(training_data, features, rate=0.1, epochs=20, smoother=10)
    if mode == "svm":
        weights, bias = support_vector_machine(training_data, features, rate=0.1, epochs=20, smoother=0.0001)

    # return weights and bias of trained perceptron
    return weights, bias


def test_classifier(tests, weights, bias, mode=""):
    num_tests = 0
    successes = 0

    if mode == 'bayes':
        for test in tests:
            num_tests +=1
            label = test[0]

            result = decide_bayes(test, weights, bias, smoother=0.1)

            if (result == 0 and label == -1) or (result == 1 and label == 1):
                successes += 1
    elif mode == 'forest':
        for test in tests:
            num_tests += 1
            label = test[0]

            result = decide_forest(test, weights)

            if (result == 0 and label == -1) or (result == 1 and label == 1):
                successes += 1
    else:
        for test in tests:
            num_tests += 1
            label = test[0]

            # print("Example: " + str(test))
            # print("Weights: " + str(weights))
            result = dot_prod(weights, test) + bias

            # print("Result: " + str(result))
            # print("Label: " + str(label))
            if result * label >= 0:
                successes += 1

    return successes / num_tests
