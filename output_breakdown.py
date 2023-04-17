import pandas as pd
import numpy as np
import sklearn
import json 
from itertools import cycle
import seaborn as sns 
import matplotlib.pyplot as plt
import os
import math

def load_sequence(file_id):
    filename = str(file_id).zfill(5)

    df = pd.read_csv('{}/train/{}/columns_1000ms.csv'.format(public_data_path, filename))
    data = df.values
    target = np.asarray(pd.read_csv('{}/train/{}/targets.csv'.format(public_data_path, filename)))[:, 2:]

    return data, target


def load_sequences(file_ids):
    x_es = []
    y_es = []

    for file_id in file_ids:
        data, target = load_sequence(file_id)

        x_es.append(data)
        y_es.append(target)

    return np.row_stack(x_es), np.row_stack(y_es)

public_data_path = 'data'

# Load the training and testing data 
train_x, train_y = load_sequences([1, 2, 3, 4, 5, 6, 7, 8])
test_x, test_y = load_sequences([9, 10])


def plot_true_outputs():
    lowest_value = float('inf')
    output_values = []

    # for i in range(train_y.shape[0]):
    #     for j in range(train_y.shape[1]):
    #         print()

    for row in train_y:
        for entry in row:
            if (math.isnan(entry) == False):
                output_values.append(entry)

    no_duplicates = list(dict.fromkeys(output_values))
    #print(no_duplicates)

    dictionary = {}
    for entry in no_duplicates:
        #dictionary.update([entry,output_values.count(entry)])
        dictionary[entry] = output_values.count(entry)
        #print("Entry: ",entry,"; ",output_values.count(entry))

    sorted_dict = dict(sorted(dictionary.items(), key=lambda item: item[1]))
    #print(sorted_dict)

    # Remove these values from our dict
    # sorted_dict.pop(1.0)

    #dict_list = list(sorted_dict)[-10]
    for key in list(sorted_dict):
        new = key.astype(str)
        if len(new.rsplit('.')[-1]) > 2:
            sorted_dict.pop(key)


    sorted_dict.pop(0.0)

    plt.rc('font',size=20)
    plt.bar(range(len(sorted_dict)), list(sorted_dict.values()), align='center')
    plt.xticks(range(len(sorted_dict)), list(sorted_dict.keys()),size=20)
    plt.yticks(size=20)
    plt.xlabel("Target.csv values")
    plt.ylabel("Count")
    plt.title("Target.csv output breakdown")

    plt.show()

plot_true_outputs()