import numpy as np

list_of_array = []

for i in range(10):
    arr = np.random.randint(low=0, high=6, size=3, dtype=int)
    list_of_array.append(arr)

for i in range(10):
    print(list_of_array[i])