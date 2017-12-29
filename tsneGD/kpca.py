# -*- coding: utf-8 -*-

import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import pdist, squareform


# ------------------------------------------------------------------------------

class Kernel:
	"""
	Base class for kernels.
	"""
	def get_kernel_matrix(self, X):
		pass


# ------------------------------------------------------------------------------

class RBF_Kernel(Kernel):
	"""
	RBF kernel class.
	"""
	def __init__(self, sigma):
		"""
		Initiate the RBF kernel with given parameters.
		:param sigma: hyperparameter sigma
		:return: self instance
		"""
		self.sigma = sigma

	def get_kernel_matrix(self, X):
		"""
		Compute RBF kernel of given data.
		:param X: Training point matrix
		:type X: array-like, shape = [n_features, n_datapoints]
		:return: corresponding kernelmatrix
		:rtype: array-like, shape = [n_datapoints, n_datapoints]
		"""
		return np.exp(-squareform(pdist(X.T)) ** 2 / (2 * self.sigma ** 2))


# ------------------------------------------------------------------------------

class Polynomial_Kernel(Kernel):
	"""
	Polynomial kernel class.
	"""
	def __init__(self, d, c=0):
		"""
		Initinate the kernel with hyperparameters. k(x,y)=(xTy + c)^d
		:param d: degree of kernel
		:param c: shift
		:return: self instance
		"""
		self.d = d
		self.c = c

	def get_kernel_matrix(self, X):
		"""
		Compute polynomial kernel of given data.
		:param X: Training point matrix
		:type X: array-like, shape = [n_features, n_datapoints]
		:return: corresponding kernelmatrix
		:rtype: array-like, shape = [n_datapoints, n_datapoints]
		"""
		return np.power(X.T.dot(X) + self.c, self.d)


# ------------------------------------------------------------------------------

class Linear_Kernel(Kernel):
	"""
	Linear kernel class.
	"""
	def __init__(self):
		pass

	def get_kernel_matrix(self, X):
		"""
		Compute linear kernel of given data.
		:param X: Training point matrix
		:type X: array-like, shape = [n_features, n_datapoints]
		:return: corresponding kernelmatrix
		:rtype: array-like, shape = [n_datapoints, n_datapoints]
		"""
		return X.T.dot(X)


# ------------------------------------------------------------------------------

class kPCA:
	"""
	Class to perform kernel PCA.
	"""
	def __init__(self, kernel, n_comps):
		"""
		Construct the kPCA object.
		:param kernel: kernel to be used
		:type kernel: Kernel
		:param n_comps: number of components to be projected on
		:type n_comps: int, long
		:return: self instance
		"""
		assert isinstance(kernel, Kernel), "use Kernel object for kernel"
		self.kernel = kernel
		assert isinstance(n_comps, (int)), "give an integer for n_comps"
		self.n_comps = n_comps
		self.alpha = None

	def fit_transform(self, X):
		"""
		Fit the data, i.e. compute eigenvectors and eigenvalues. And compute the projections.
		:param X: data matrix
		:type X: array-like, shape = [n_features, n_datapoints]
		:return: corresponding projections
		:rtype: array-like, shape = [n_proj_comps, n_datapoints]
		"""
		self.X = X
		self._center()

		K = self.kernel.get_kernel_matrix(X)

		m = K.shape[0]
		self.eig_val, self.eig_vec = eigh(K, eigvals=(m - self.n_comps, m - 1))
		self.alpha = np.fliplr(np.divide(self.eig_vec, np.sqrt(self.eig_val)))

		return K.dot(self.alpha).T

	def _center(self):
		"""
		Center the data
		"""
		m = np.mean(self.X, axis=1)
		for c in self.X.T:
			c -= m


# ------------------------------------------------------------------------------

if __name__ == "__main__":

	rbf = RBF_Kernel(1.0)
	lin = Linear_Kernel()
	pol = Polynomial_Kernel(3)

	X = np.diag([1, 2, 3])
	#X = np.vstack((X, X)) + np.random.rand(6, 3)

	for k in (rbf, pol, lin):
		p = kPCA(k, 2)
		print(p.fit_transform(X))