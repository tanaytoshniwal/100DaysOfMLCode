# Vectorization
Using Numpy Library of Python, I implemented a simple dot product of two Numpy matrices, one using Vectorization, and another using explicit for loop:
```python
import time
import numpy as np
a = np.random.rand(3000000)
b = np.random.rand(3000000)
init = time.time()
c = np.dot(a,b)
fin = time.time()
print(c)
print("time: " + str(1000 * (fin-init))+"ms")
c = 0
init = time.time()
for i in range(3000000):
    c += a[i]*b[i]
fin = time.time()
print(c)
print("time: " + str(1000 * (fin-init))+"ms")
```
