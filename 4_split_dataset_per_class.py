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
def make_count_dic(_labels):
    count_dic = {}
    for y in _labels:
        try:
            count_dic[y] += 1
        except KeyError:
            count_dic[y] = 1
    return count_dic

def make_class_dic(inputs, labels):
    class_dic = {}
    for x, y in zip(inputs, labels):
        try:
            class_dic[y].append(x)
        except KeyError:
            class_dic[y] = [x]
    return class_dic

def bar_show(count_dic, title):
    _labels = sorted(count_dic.keys())

    labels = [str(label) for label in _labels]
    counts = [count_dic[label] for label in _labels]

    plt.clf()
    plt.title(title)
    plt.bar(labels, counts)
    plt.show()

class_dic = make_class_dic(inputs, labels)

# 50 [array([5.1, 3.5, 1.4, 0.2]), array([4.9, 3. , 1.4, 0.2]), array([4.7, 3.2, 1.3, 0.2]), array([4.6, 3.1, 1.5, 0.2]), array([5. , 3.6, 1.4, 0.2])]
# 50 [array([7. , 3.2, 4.7, 1.4]), array([6.4, 3.2, 4.5, 1.5]), array([6.9, 3.1, 4.9, 1.5]), array([5.5, 2.3, 4. , 1.3]), array([6.5, 2.8, 4.6, 1.5])]
# 50 [array([6.3, 3.3, 6. , 2.5]), array([5.8, 2.7, 5.1, 1.9]), array([7.1, 3. , 5.9, 2.1]), array([6.3, 2.9, 5.6, 1.8]), array([6.5, 3. , 5.8, 2.2])]
# print(len(class_dic[0]), class_dic[0][:5])
# print(len(class_dic[1]), class_dic[1][:5])
# print(len(class_dic[2]), class_dic[2][:5])

train_inputs = []
train_labels = []
test_inputs = []
test_labels = []

for label in list(class_dic.keys()):
    inputs_per_class = np.asarray(class_dic[label])

    length = len(inputs_per_class)
    index_list = np.arange(length)
    np.random.shuffle(index_list)
    
    train_length = int(length * 0.8)
    train_index_list = index_list[:train_length]
    test_index_list = index_list[train_length:]

    train_inputs += inputs_per_class[train_index_list].tolist()
    train_labels += (np.ones(train_length, dtype=np.int32) * label).tolist()

    test_inputs += inputs_per_class[test_index_list].tolist()
    test_labels += (np.ones(length - train_length, dtype=np.int32) * label).tolist()

# train_count_dic = make_count_dic(train_labels) 
# test_count_dic = make_count_dic(test_labels)

# {0: 50, 1: 50, 2: 20}
# {2: 30}
# print(train_count_dic)
# print(test_count_dic)

# bar_show(train_count_dic, '# train set')
# bar_show(test_count_dic, '# test set')

df = pd.DataFrame(data=train_inputs, columns=feature_names[:-2])
df['label'] = train_labels
df.to_csv('./data/train.csv')

df = pd.DataFrame(data=test_inputs, columns=feature_names[:-2])
df['label'] = test_labels
df.to_csv('./data/test.csv')