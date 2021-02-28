
import numpy as np
import matplotlib.pyplot as plt
from svm import SVM

##
##	Generate Data
##
classA = np.concatenate((
	np.random.randn(20, 2) * 0.2 + [1 , 1],
	np.random.randn(20, 2) * 0.2 + [-4, 0]
	))

classB = np.random.randn(20, 2) * 0.2 + [-1 , -1.25]

X = np.concatenate((classA, classB))
y = np.concatenate(
	(np.ones(classA.shape[0]),
		-np.ones(classB.shape[0])))

Npts = X.shape[0] # Number of rows (samples)

plt.style.use('seaborn')
plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.', label="class A")
plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.', label="class B")

##
##	Choose Experiment
##
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# 1. Linear Kernel
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
svm = SVM()

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# 2. Polynomail Kernel
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#svm = SVM('poly', C = 2, degree=2)

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# 3. Radial Basis Function
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#svm = SVM('rbf', C = 2, var=0.4)


svm.fit(X,y)
svm.visualize()
