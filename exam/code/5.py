import numpy as np
import pandas as pd

data = pd.read_csv('../heart_simplified_RandomForest.csv')

# Subtask a
def entropy(data):
    count = np.bincount(data["HeartDisease"])
    if len(count) < 2: # If there is only one class the leaf is pure
        return 0
    pos = count[1] / np.sum(count) # Heart disease
    neg = count[0] / np.sum(count) # No heart disease
    
    if pos == 0 or neg == 0: # To avoid runtime divide by zero error
        return 0
    
    entropy = (-pos) * np.log2(pos) - neg * np.log2(neg)

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
quality_of_threshold(data, np.mean(data["Age"]), "Age")
quality_of_threshold(data, np.mean(data["RestingBP"]), "RestingBP")
quality_of_threshold(data, np.mean(data["Cholesterol"]), "Cholesterol")
quality_of_threshold(data, np.mean(data["MaxHR"]), "MaxHR")

# Subtask b
def find_threshold(data, column_name):
    thresholds = np.unique(data[column_name])
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