import numpy as np
import matplotlib.pyplot as plt

'''
[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47
 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71
 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95
 96 97 98 99]
'''
vector = np.arange(100)

# np.random.seed(0)
# np.random.shuffle(vector)

'''
[34 70 45 12 42 31 27 14 17 73 21 13 10 57  4 35  1 79 81 72 65  9 11 33
 36 74  6 82 62 56 38 49 19 91 54 83 86 68 64 97 20 93 32  5 48  2 23  7
 50 61 71 16  8 99 55 77  0 25 46  3 28 26 44 22 39 75 18 63 66 59 80 92
 47 95 90 29 41 85 52 53]
[43 76 60 84 69 40 24 98 88 94 96 30 15 51 67 89 78 37 58 87]
'''
# train_x = vector[:80]
# test_x = vector[80:]

# plt.plot(train_x)
# plt.plot(test_x)
# plt.show()

# print(train_x)
# print(test_x)

vector = np.arange(100).tolist()
print(vector[0], vector[1], vector[4], vector[5])

print(vector[[0, 1, 4, 5]])
