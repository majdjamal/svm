
"""svm.py: Support Vector Machine with linear and non-linear function."""

__author__ = "Majd Jamal"

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class SVM:
	"""
	Support Vector Machine
	"""
	def __init__(self, kernel = 'linear', C=1, degree = 2, var = 0.5):

		#Parameters
		self.kernel = kernel 	#Kernel function
		self.C = C 				#Slack variable
		self.degree = degree	#Degree in the polynomial function
		self.var = var 			#Variance in the RBF

		#Global variables
		self.patterns = None
		self.targets = None
		self.Npts = None
		self.alpha = None
		self.b = None
		self.nonZeros = None
		self.kernelMTX = None

	def linear(self, a, b):
		""" Linear kernel function
		:param a: Vector A
		:param b: Vector B
		:return: Dot product between A and B
		"""
		return np.dot(a, b)

	def poly(self, a, b):
		""" Polynomial kernel function
		:param a: Vector A
		:param b: Vector B
		:return: Function values
		"""
		return (np.dot(a, b) + 1) ** self.degree

	def rbf(self, p1, p2):
		""" Radial Basis Function kernel
		:param p1: Vector A
		:param p2: Vector B
		:return: Function values
		"""
		numerator = np.subtract(p1, p2)
		numerator = np.square(np.linalg.norm(numerator))
		denomintaor = 2 * np.square(self.var)
		return np.exp(-(numerator/denomintaor))

	def kernelMatrix(self, X, y):
		""" Compute the kernel matrix
		:param X: patterns
		:param y: targets
		:return: Kernel Matrix
		"""
		matrix = np.zeros((self.Npts,self.Npts))

		for i in range(self.Npts):
			for j in range(self.Npts):

				if self.kernel == 'linear':
					func = self.linear(X[i], X[j])
				elif self.kernel == 'poly':
					func = self.poly(X[i], X[j])
				elif self.kernel == 'rbf':
					func = self.rbf(X[i], X[j])

				matrix[i][j] = y[i]*y[j]*func

		return matrix

	def dualProblem(self, alpha):
		""" Computes the Dual Problem. This function
			implements equation 4 from the instruction
			file.
		:param alpha: weights
		:return: Function values
		"""
		new_arr = np.array([alpha])
		summation = new_arr @ self.kernelMTX @ new_arr.T * 0.5

		return summation[0] - np.sum(alpha)

	def zerofun(self, alpha):
		""" Equality constraint. This function
			implements the equation 10 from
			the instruction file.
		:param alpha: weights
		:return: Function values
		"""
		return np.dot(alpha, self.targets)

	def nonZero(self):
		""" Uses alpha to find support vectors.
		:return: Indicies of support vectors, i.e. where
				 alpha is non-zero.
		"""
		return np.where(self.alpha > 1e-5)[0]


	def calcB(self):
		""" Computes B-values, using
			equation 7 from the instruction
			file.
		"""
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

	def visualize(self):
		""" Visualize the decision boundary.
		"""
		xgrid=np.linspace(-5, 5)
		ygrid=np.linspace(-4, 4)

		grid=np.array([[self.indicator(x, y)
			for x in xgrid]
			for y in ygrid])

		plt.style.use('seaborn')
		plt.contour(xgrid, ygrid, grid,
			(-1.0, 0.0, 1.0),
			colors=('red', 'black', 'blue'),
			linewidths=(1, 3, 1))

		plt.xlabel('x1')
		plt.ylabel('x2')
		plt.axis('equal') # Force same scale on both axes plt.savefig(’svmplot.pdf’) # Save a copy in a file plt .show() # Show the plot on the screen
		plt.legend()
		plt.show() # Show the plot on the screen


	def indicator(self, x, y):
		""" Indicator. This function takes
			a data point and investigates at which
			side of the decision boundary it is located.
			It is an implementation of equation 6
			from the instruction file.
		:param x: The x-coordinate of a data point
		:param y: The y-coordinate of a data point
		:return indication: classification
		"""
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

	def fit(self, X, y):
		""" Train the Support Vector Machine.
			This function uses SciPy optimze to
			find the optimal alpha values.
		:param X: Data Matrix. Shape is (Npts, Ndim = 2)
		:param y: Labels
		"""
		self.Npts = X.shape[0]
		self.patterns = X
		self.targets = y
		self.kernelMTX = self.kernelMatrix(X, y)

		self.C = 1
		B = [(0, self.C) for b in range(self.Npts)]
		XC = {'type':'eq', 'fun':self.zerofun}

		ret = minimize(self.dualProblem, np.random.normal(self.Npts), bounds=B, constraints=XC)
		self.alpha = ret['x']
		self.nonZeros = self.nonZero()
		self.calcB()

	def predict(self, X):
		""" Classifies data point-s.
		:param X: Data Matrix
		:return: label-s
		"""
		if X[0].size == 1:

			pred = self.indicator(X[0], X[1])
			return np.where(pred > 0, 1, -1)

		else:

			Npts, _ = X.shape

			predictions = np.zeros(Npts)

			for i in range(Npts):
				pred = self.indicator(X[i][0], X[i][1])
				predictions[i] = np.where(pred > 0, 1, -1)

			return predictions
