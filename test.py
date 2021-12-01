import numpy as np
import time

s = time.time()
a = np.load("messidor.npy")
print(a.shape)
e = time.time()
print("load durations is ", e -s)