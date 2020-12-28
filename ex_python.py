
a = [1, 2, 3, 4, 5]

# for i in range(len(a)):
#     a[i] += 1

a = [a[i] + 1 for i in range(len(a))]
print(a)