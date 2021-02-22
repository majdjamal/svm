
"""svm.py: Support Vector Machine with linear and non-linear function."""

__author__ = "Majd Jamal"

import numpy as np
from scipy.optimize import minimize 
import matplotlib.pyplot as plt

class SVM:

	def __init__(self, kernel = 'linear', C=1, degree = 2, var = 0.5):

		#Parameters
		self.kernel = kernel
		self.degree = degree
		self.var = var
		self.C = 1

		#Global variables
		self.patterns = None
		self.targets = None
		self.Npts = None
		self.alpha = None
		self.b = None
		self.nonZeros = None
		self.kernelMTX = None

	def linear(self, a, b):
		return np.dot(a, b)

	def poly(self, a, b):
		return (np.dot(a, b) + 1) ** self.degree

	def rbf(self, p1, p2):
		numerator = np.subtract(p1, p2)
		numerator = np.linalg.norm(numerator) ** self.var
		denomintaor = 2 * np.square(smooth)
		return np.exp(-(numerator/denomintaor))

	def kernelMatrix(self, X, y):

		matrix = np.zeros((Npts,Npts))

		for i in range(Npts):
			for j in range(Npts):

				if self.kernel == 'linear':
					func = self.linear(X[i], X[j])
				elif self.kernel == 'poly':
					func = self.poly(X[i], X[j])
				elif self.kernel == 'rbf':
					func = self.rbf(X[i], X[j])

				matrix[i][j] = y[i]*y[j]*func

		return matrix

	def dualProblem(self, alpha):

		new_arr = np.array([alpha])
		summation = new_arr @ self.kernelMTX @ new_arr.T * 0.5

		return summation[0] - np.sum(alpha)

	def zerofun(self, alpha):
		return np.dot(alpha, self.targets)

	def nonZero(self):
		return np.where(self.alpha > 1e-5)[0]
	
	#alpha as input
	def calcB(self):

		b = 0

		for i in self.nonZeros:
			if self.alpha[i] < self.C:
				for j in range(self.Npts):

					if self.kernel == 'linear':
						func = self.linear(self.patterns[i], self.patterns[j])
					elif self.kernel == 'poly':
						func = self.poly(self.patterns[i], self.patterns[j])
					elif self.kernel == 'rbf':
						func = self.rbf(self.patterns[i], self.patterns[j])
				
					b += self.alpha[j]*self.targets[j]*func	
				
				b -= self.targets[i]

		self.b = b/self.nonZeros.shape[0]

	def fit(self, X, y):

		self.Npts = X.shape[0]
		self.patterns = X
		self.targets = y
		self.kernelMTX = self.kernelMatrix(X, y)

		self.C = 1
		B = [(0, self.C) for b in range(Npts)]
		XC = {'type':'eq', 'fun':self.zerofun}

		ret = minimize(self.dualProblem, np.random.normal(Npts), bounds=B, constraints=XC)
		self.alpha = ret['x']
		self.nonZeros = self.nonZero()
		self.calcB()

	def predict(self, x, y, decision_boundary=False):

		ind = 0
		for i in self.nonZeros:

			if self.kernel == 'linear':
				func = self.linear((x, y), self.patterns[i])
			elif self.kernel == 'poly':
				func = self.poly((x, y), self.patterns[i])
			elif self.kernel == 'rbf':
				func = self.rbf((x, y), self.patterns[i])
		
			ind += self.alpha[i]*self.targets[i]*func

		indication = ind - self.b
		
		return indication

classA = np.random.randn(20, 2) * 0.2 + [1 , 1]
classB = np.random.randn(20, 2) * 0.2 + [-1 , -1]

X = np.concatenate((classA, classB)) 
y = np.concatenate(
	(np.ones(classA.shape[0]), 
		-np.ones(classB.shape[0])))

Npts = X.shape[0] # Number of rows (samples)

plt.style.use('seaborn')
plt.plot([p[0] for p in classA], [p[1] for p in classA], 'b.', label="class A") 
plt.plot([p[0] for p in classB], [p[1] for p in classB], 'r.', label="class B")

svm = SVM('linear')
svm.fit(X,y)

xgrid=np.linspace(-5, 5) 
ygrid=np.linspace(-4, 4)

grid=np.array([[svm.predict(x, y, decision_boundary = True) 
	for x in xgrid] 
	for y in ygrid])

plt.contour(xgrid, ygrid, grid, 
	(-1.0, 0.0, 1.0), 
	colors=('red', 'black', 'blue'), 
	linewidths=(1, 3, 1))

plt.xlabel('x1')
plt.ylabel('x2')
plt.axis('equal') # Force same scale on both axes plt.savefig(’svmplot.pdf’) # Save a copy in a file plt .show() # Show the plot on the screen
plt.legend()
plt.show() # Show the plot on the screen
