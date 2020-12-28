from sklearn.datasets import load_iris

dataset = load_iris()

# <class 'sklearn.utils.Bunch'>
# print(type(dataset))

# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
# ['setosa' 'versicolor' 'virginica']
# print(dataset.feature_names)
# print(dataset.target_names)

# print(dataset.data)   # X
# print(dataset.target) # Y

import pandas as pd
df = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
df['label'] = dataset.target

class_dic = {}
for class_index, class_name in enumerate(dataset.target_names):
    class_dic[class_index] = class_name

# print(class_dic[0]) # -> 
# print(class_dic[1]) # -> 
# print(class_dic[2]) # -> 

df['class_names'] = df['label'].map(class_dic)
df.to_csv('./data/dataset.csv')

