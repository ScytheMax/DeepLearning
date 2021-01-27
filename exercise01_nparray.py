#[1] pages 1-6
#[2] folder 01_foundations\Code.ipynb "Illustration of Python lists vs. Numpy arrays"

import numpy as np

print("Python list operations:")
a = [1,2,3]
b = [4,5,6]
c = [99, 98]
print("a =", a)
print("b =", b)
print("c =", c)
print("a + b =", a + b)
try:
    print(a * b)
except:
    print("a * b has no meaning for Python lists")
try:
    print(a + c)
except:
    print("a + c has no meaning for Python lists")

print()
print("Numpy array operations:")
a = np.array([1,2,3])
b = np.array([4,5,6])
c = np.array([99,98])
print("a =", a)
print("b =", b)
print("c =", c)
print("a + b =", a + b)
print("a * b =", a * b)
try:
    print("a + c =", a + c)
except:
    print("a + c has no meaning for np arrays")

a = np.array([[1,2,3],
              [4,5,6]]) 
b = np.array([10,20,30])
c = np.array([10,20])
d = np.array([[10],
              [20]])
print("a =", a)
print("b =", b)
print("c =", c)
print("d =", d)
print("a + b =\n", a + b)
print("a + d =\n", a + d)
try:
    print("a + c =", a + c)
except:
    print("a + c has no meaning for np arrays")

print('a.sum(axis=0) =', a.sum(axis=0))
print('a.sum(axis=1) =', a.sum(axis=1))

