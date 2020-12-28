import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('./data/dataset.csv')
del df['Unnamed: 0']

print(df.columns)

sns.pairplot(df, hue="label", height=3)
plt.show()

