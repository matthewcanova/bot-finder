import feature_processor
import classifier


def load_file(file_name):
    with open(file_name) as f:
        data = f.read()
    return data


def process_features(features):
    new_features = []

    for example in features:
        new_example = {0: example[0]}
        marker = feature_processor.feat_1(example, new_example, 1)
        marker = feature_processor.feat_2(example, new_example, marker)
        marker = feature_processor.feat_3(example, new_example, marker)
        marker = feature_processor.feat_7(example, new_example, marker)
        marker = feature_processor.feat_8(example, new_example, marker)
        marker = feature_processor.feat_9(example, new_example, marker)
        marker = feature_processor.feat_10(example, new_example, marker)
        marker = feature_processor.feat_11(example, new_example, marker)
        marker = feature_processor.feat_12(example, new_example, marker)
        marker = feature_processor.feat_13(example, new_example, marker)
        marker = feature_processor.feat_14(example, new_example, marker)
        marker = feature_processor.feat_15(example, new_example, marker)
        feature_processor.feat_16(example, new_example, marker)
        new_features.append(new_example)

    return new_features


#####################################
# DATA IMPORT AND FEATURE EXTRACTION
#####################################

# Load files
train_data = load_file("DatasetRetry/data-splits/data.train")

train_data_id = load_file("DatasetRetry/data-splits/data.train.id")

test_data = load_file("DatasetRetry/data-splits/data.test")

test_data_id = load_file("DatasetRetry/data-splits/data.test.id")

eval_data = load_file("DatasetRetry/data-splits/data.eval.anon")

eval_data_id = load_file("DatasetRetry/data-splits/data.eval.id")
eval_data_id = eval_data_id.splitlines()

# Process data with original 16 features and real values
train_data = classifier.process_data(train_data)

test_data = classifier.process_data(test_data)

eval_data = classifier.process_data(eval_data)

# Process data into binary features
train_feature_data = process_features(train_data)

test_feature_data = process_features(test_data)

eval_feature_data = process_features(eval_data)


################
# DECISION TREE
################

# tree = decision_tree.train_tree(train_feature_data, depth=999)
#
# results = decision_tree.test_tree(tree, test_feature_data)
#
# print(results)
#
# # Test the classifier with test data
# with open("eval.tree.results", 'w') as file:
#     file.write('Id,Prediction\n')
#     for index in range(len(eval_feature_data)):
#         file.write(str(eval_data_id[index]) +
#                    ',' +
#                    str(decision_tree.run_tree(tree, eval_feature_data[index])) +
#                    '\n')


###############################
# PERCEPTRON LINEAR CLASSIFIER
###############################

# # Train a classifier on the data
# weights, bias = classifier.train_classifier(train_feature_data, mode='dynamic')
#
# success_rate = classifier.test_classifier(test_feature_data, weights, bias)
#
# print(success_rate)
#
# # Test the classifier with test data
# with open("eval.perceptron.results", 'w') as file:
#     file.write('Id,Prediction\n')
#     for index in range(len(eval_feature_data)):
#         file.write(str(eval_data_id[index]) +
#                    ',' +
#                    str(classifier.run_perceptron(eval_feature_data[index], weights, bias)) +
#                    '\n')


##############
# NAIVE BAYES
##############

# # Train a classifier on the data
# weights, bias = classifier.train_classifier(train_feature_data, mode='bayes')
#
# success_rate = classifier.test_classifier(test_feature_data, weights, bias, mode='bayes')
#
# print(success_rate)
#
# # Test the classifier with test data
# with open("eval.bayes.results", 'w') as file:
#     file.write('Id,Prediction\n')
#     for index in range(len(eval_feature_data)):
#         file.write(str(eval_data_id[index]) +
#                    ',' +
#                    str(classifier.decide_bayes(eval_feature_data[index], weights, bias)) +
#                    '\n')


#################
# BAGGES FORESTS
#################

# Train a classifier on the data
forest, bias = classifier.train_classifier(train_feature_data, mode='forest')

success_rate = classifier.test_classifier(test_feature_data, forest, 0, mode='forest')

print(success_rate)

# Test the classifier with test data
with open("eval.forest.results", 'w') as file:
    file.write('Id,Prediction\n')
    for index in range(len(eval_feature_data)):
        file.write(str(eval_data_id[index]) +
                   ',' +
                   str(classifier.decide_forest(eval_feature_data[index], forest)) +
                   '\n')


#########################
# SUPPORT VECTOR MACHINE
#########################

# classifier.cross_validate(train_feature_data, mode='svm')

# # Train a classifier on the data
# weights, bias = classifier.train_classifier(train_feature_data, mode='svm')
#
# success_rate = classifier.test_classifier(test_feature_data, weights, bias, mode='svm')
#
# print(success_rate)
#
# # Test the classifier with test data
# with open("eval.svm.results", 'w') as file:
#     file.write('Id,Prediction\n')
#     for index in range(len(eval_feature_data)):
#         file.write(str(eval_data_id[index]) +
#                    ',' +
#                    str(classifier.run_perceptron(eval_feature_data[index], weights, bias)) +
#                    '\n')


# #########################
# # LOGISTIC REGRESSION
# #########################
#
# classifier.cross_validate(train_feature_data, mode='log')
#
# # Train a classifier on the data
# weights, bias = classifier.train_classifier(train_feature_data, mode='log')
#
# success_rate = classifier.test_classifier(test_feature_data, weights, bias, mode='log')
#
# print(success_rate)
#
# # Test the classifier with test data
# with open("eval.log.results", 'w') as file:
#     file.write('Id,Prediction\n')
#     for index in range(len(eval_feature_data)):
#         file.write(str(eval_data_id[index]) +
#                    ',' +
#                    str(classifier.run_perceptron(eval_feature_data[index], weights, bias)) +
#                    '\n')
