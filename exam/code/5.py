import pandas as pd
import math

data = pd.read_csv('../heart_simplified_RandomForest.csv')

# Helper functions
def count_occurences(data):
    count_array = {}

    for occurence in data:
        if occurence in count_array:
            count_array[occurence] += 1
        else:
            count_array[occurence] = 1
    return count_array

# Subtask a
def entropy(data):
    count = count_occurences(data["HeartDisease"])
    if len(count) < 2: # If there is only one class the leaf is pure
        return 0
    
    sum_count = 0
    for key in count:
        sum_count += count[key]
    pos = count[1] / sum_count # Heart disease
    neg = count[0] / sum_count # No heart disease
    
    entropy = (-pos) * math.log2(pos) - neg * math.log2(neg)

    return entropy

def quality_of_threshold(data, threshold, column_name):
    # Split the data into two parts
    data_left = data[data[column_name] <= threshold]
    data_right = data[data[column_name] > threshold]
    information_gain = entropy(data) - (len(data_left) / len(data)) * entropy(data_left) - (len(data_right) / len(data)) * entropy(data_right)
    print("    For feature", column_name, "and the threshold", threshold, "(entropy) quality is:", information_gain)
    return information_gain

print("========================================")
print("Subtask a")
print("========================================")
quality_of_threshold(data, sum(data["Age"]) / len(data["Age"]), "Age")
quality_of_threshold(data, sum(data["RestingBP"])  / len(data["RestingBP"]), "RestingBP")
quality_of_threshold(data, sum(data["Cholesterol"])  / len(data["Cholesterol"]), "Cholesterol")
quality_of_threshold(data, sum(data["MaxHR"])  / len(data["MaxHR"]), "MaxHR")

# Subtask b
def find_threshold(data, column_name):
    thresholds = []
    for value in data[column_name]:
        if value not in thresholds:
            thresholds.append(value)

    best_threshold = None
    best_quality = 0
    for threshold in thresholds:
        quality = quality_of_threshold(data, threshold, column_name)
        if quality > best_quality:
            best_quality = quality
            best_threshold = threshold
    print("  The best threshold for feature", column_name, "is", best_threshold, "with quality", best_quality)
    return best_threshold, best_quality

print("========================================")
print("Subtask b")
print("========================================")
find_threshold(data, "Age")

# Subtask c
def find_feature(data, features):
    best_feature = None
    best_quality = 0
    best_threshold = None
    for column_name in data.columns:
        if column_name not in features:
            continue
        threshold, quality = find_threshold(data, column_name)
        print
        if quality > best_quality:
            best_quality = quality
            best_threshold = threshold
            best_feature = column_name
    print("The best feature is", best_feature, "with threshold", best_threshold, "and quality", best_quality)
    return best_feature, best_threshold, best_quality

print("========================================")
print("Subtask c")
print("========================================")
find_feature(data, ["Age", "RestingBP", "Cholesterol", "MaxHR"])