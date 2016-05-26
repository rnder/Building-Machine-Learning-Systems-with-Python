### **Learning NumPy**

>>> import numpy
>>> numpy.version.full_version
>>> from numpy import *

>>> import numpy as np
>>> a = np.array([0,1,2,3,4,5])
>>> a
array([0, 1, 2, 3, 4, 5])
>>> a.ndim
1
>>> a.shape
(6,)

>>> b = a.reshape((3,2))
>>> b
array([[0, 1],
[2, 3],
[4, 5]])
>>> b.ndim
2
>>> b.shape
(3, 2)

>>> b[1][0] = 77
>>> b
array([[ 0, 1],
[77, 3],
[ 4, 5]])
>>> a
array([ 0, 1, 77, 3, 4, 5])

>>> c = a.reshape((3,2)).copy()
>>> c
array([[ 0, 1],
[77, 3],
[ 4, 5]])
>>> c[0][0] = -99
>>> a
array([ 0, 1, 77, 3, 4, 5])
>>> c
array([[-99, 1],
[ 77, 3],
[ 4, 5]])

>>> d = np.array([1,2,3,4,5])
>>> d*2
array([ 2, 4, 6, 8, 10])

>>> d**2
array([ 1, 4, 9, 16, 25])

>>> [1,2,3,4,5]*2
[1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
>>> [1,2,3,4,5]**2

### **Indexing**

>>> a[np.array([2,3,4])]
array([77, 3, 4])

>>> a>4
array([False, False, True, False, False, True], dtype=bool)
>>> a[a>4]
array([77, 5])

>>> a[a>4] = 4
>>> a
array([0, 1, 4, 3, 4, 4])

>>> a.clip(0,4)
array([0, 1, 4, 3, 4, 4])

### **Handling nonexisting values**
>>> c = np.array([1, 2, np.NAN, 3, 4]) # let's pretend we have read this
from a text file
>>> c
array([ 1., 2., nan, 3., 4.])
>>> np.isnan(c)
array([False, False, True, False, False], dtype=bool)
>>> c[~np.isnan(c)]
array([ 1., 2., 3., 4.])
>>> np.mean(c[~np.isnan(c)])
2.5

### **Comparing the runtime**

import timeit
normal_py_sec = timeit.timeit('sum(x*x for x in range(1000))',
number=10000)
naive_np_sec = timeit.timeit(
'sum(na*na)',
setup="import numpy as np; na=np.arange(1000)",
number=10000)
good_np_sec = timeit.timeit(
'na.dot(na)',
setup="import numpy as np; na=np.arange(1000)",
number=10000)
print("Normal Python: %f sec" % normal_py_sec)
print("Naive NumPy: %f sec" % naive_np_sec)
print("Good NumPy: %f sec" % good_np_sec)
Normal Python: 1.050749 sec
Naive NumPy: 3.962259 sec
Good NumPy: 0.040481 sec

>>> a = np.array([1,2,3])
>>> a.dtype
dtype('int64')

>>> np.array([1, "stringy"])
array(['1', 'stringy'], dtype='<U7')
>>> np.array([1, "stringy", set([1,2,3])])
array([1, stringy, {1, 2, 3}], dtype=object)

### **Learning SciPy**

>>> import scipy, numpy
>>> scipy.version.full_version
0.14.0
>>> scipy.dot is numpy.dot
True

### **Our first (tiny) application of machine learning**
### **Reading in the data**

>>> import scipy as sp
>>> data = sp.genfromtxt("web_traffic.tsv", delimiter="\t"

>>> print(data[:10])
[[ 1.00000000e+00 2.27200000e+03]
[ 2.00000000e+00 nan]
[ 3.00000000e+00 1.38600000e+03]
[ 4.00000000e+00 1.36500000e+03]
[ 5.00000000e+00 1.48800000e+03]
[ 6.00000000e+00 1.33700000e+03]
[ 7.00000000e+00 1.88300000e+03]
[ 8.00000000e+00 2.28300000e+03]
[ 9.00000000e+00 1.33500000e+03]
[ 1.00000000e+01 1.02500000e+03]]
>>> print(data.shape)
(743, 2)

### **Preprocessing and cleaning the data**

x = data[:,0]
y = data[:,1]

>>> sp.sum(sp.isnan(y))
8

>>> x = x[~sp.isnan(y)]
>>> y = y[~sp.isnan(y)]

>>> import matplotlib.pyplot as plt
>>> # plot the (x,y) points with dots of size 10
>>> plt.scatter(x, y, s=10)
>>> plt.title("Web traffic over the last month")
>>> plt.xlabel("Time")
>>> plt.ylabel("Hits/hour")
>>> plt.xticks([w*7*24 for w in range(10)],
['week %i' % w for w in range(10)])
>>> plt.autoscale(tight=True)
>>> # draw a slightly opaque, dashed grid
>>> plt.grid(True, linestyle='-', color='0.75')
>>> plt.show()

### **Choosing the right model and learning algorithm**
def error(f, x, y):
return sp.sum((f(x)-y)**2)

fp1, residuals, rank, sv, rcond = sp.polyfit(x, y, 1, full=True)

>>> print("Model parameters: %s" % fp1)
Model parameters: [ 2.59619213 989.02487106]
>>> print(residuals)
[ 3.17389767e+08]

f(x) = 2.59619213 * x + 989.02487106.

>>> f1 = sp.poly1d(fp1)
>>> print(error(f1, x, y))
317389767.34

fx = sp.linspace(0,x[-1], 1000) # generate X-values for plotting
plt.plot(fx, f1(fx), linewidth=4)
plt.legend(["d=%i" % f1.order], loc="upper left")

### **Towards some advanced stuff**
>>> f2p = sp.polyfit(x, y, 2)
>>> print(f2p)
array([ 1.05322215e-02, -5.26545650e+00, 1.97476082e+03])
>>> f2 = sp.poly1d(f2p)
>>> print(error(f2, x, y))
179983507.878

f(x) = 0.0105322215 * x**2 - 5.26545650 * x + 1974.76082

### **Stepping back to go forward – another look at our data**
inflection = 3.5*7*24 # calculate the inflection point in hours
xa = x[:inflection] # data before the inflection point
ya = y[:inflection]
xb = x[inflection:] # data after
yb = y[inflection:]

fa = sp.poly1d(sp.polyfit(xa, ya, 1))
fb = sp.poly1d(sp.polyfit(xb, yb, 1))

fa_error = error(fa, xa, ya)
fb_error = error(fb, xb, yb)
print("Error inflection=%f" % (fa_error + fb_error))
Error inflection=132950348.197616

### **Training and testing**
Error d=1: 6397694.386394
Error d=2: 6010775.401243
Error d=3: 6047678.658525
Error d=10: 7037551.009519
Error d=53: 7052400.001761

### **Answering our initial question**
>>> fbt2 = sp.poly1d(sp.polyfit(xb[train], yb[train], 2))
>>> print("fbt2(x)= \n%s" % fbt2)
fbt2(x)=
2
0.086 x - 94.02 x + 2.744e+04
>>> print("fbt2(x)-100,000= \n%s" % (fbt2-100000))
fbt2(x)-100,000=
2
0.086 x - 94.02 x - 7.256e+04
>>> from scipy.optimize import fsolve
>>> reached_max = fsolve(fbt2-100000, x0=800)/(7*24)
>>> print("100,000 hits/hour expected at week %f" % reached_max[0])
