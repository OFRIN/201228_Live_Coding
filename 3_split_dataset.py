import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('./data/dataset.csv')
del df['Unnamed: 0']

feature_names = df.columns

# 1. convert numpy
inputs = df.iloc[:, :-2].values
labels = df.iloc[:, -2].values
class_names = df.iloc[:, -1].values

# [5.1 3.5 1.4 0.2] 0
# print(inputs[0], labels[0])

# for input, label, class_name in zip(inputs, labels, class_names):
#     for name, x in zip(feature_names, input):
#         print(name, x)

#     print('label = {}, class_name = {}'.format(label, class_name))
#     print()

# 2. split train set, test set
length = len(inputs)
index_list = np.arange(length)

np.random.seed(0)
np.random.shuffle(index_list)

train_length = int(length * 0.8)
train_index_list = index_list[:train_length]
test_index_list = index_list[train_length:]

train_inputs = inputs[train_index_list]
train_labels = labels[train_index_list]

test_inputs = inputs[test_index_list]
test_labels = labels[test_index_list]

def make_count_dic(_labels):
    count_dic = {}
    for y in _labels:
        try:
            count_dic[y] += 1
        except KeyError:
            count_dic[y] = 1
    return count_dic

def bar_show(count_dic, title):
    _labels = sorted(count_dic.keys())

    labels = [str(label) for label in _labels]
    counts = [count_dic[label] for label in _labels]

    plt.clf()
    plt.title(title)
    plt.bar(labels, counts)
    plt.show()

train_count_dic = make_count_dic(train_labels) 
test_count_dic = make_count_dic(test_labels)

# {0: 50, 1: 50, 2: 20}
# {2: 30}
print(train_count_dic)
print(test_count_dic)

bar_show(train_count_dic, '# train set')
bar_show(test_count_dic, '# test set')