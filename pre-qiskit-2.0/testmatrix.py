import numpy as np
import random
from random import sample

rows = 3
cols = 3

offset = 4

arr = np.zeros((rows, cols))

samples = list(np.arange(0, rows*cols))
sample_list = sample(samples, k=rows*cols)
real = random.choice(sample_list)

for i in range(0, rows):
    for j in range(0, cols):
        arr[i, j] = sample_list[i*cols + j] + offset


print(arr)