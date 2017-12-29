#  coding: utf-8
#  GD_tsne.py
#
# Implementation of t-SNE in Python. The implementation was tested on Python 2.7.10, and it requires a working
# installation of NumPy. The implementation comes with an example on the MNIST dataset. In order to plot the
# results of this example, a working installation of matplotlib is required.
#
# The example can be run by executing: `ipython GD_tsne.py`
#
#
#  Created by Laurens van der Maaten on 20-12-08.
#  Copyright (c) 2008 Tilburg University. All rights reserved.


import numpy as np
import pylab as Plot
from numpy import array, asarray, inf, zeros, minimum, diagonal, newaxis
import kpca
import scipy.sparse.csgraph

INF = 10.0

def Hbeta(D = np.array([]), beta = 1.0):
	"""Compute the perplexity and the P-row for a specific value of the precision of a Gaussian distribution."""

	# Compute P-row and corresponding perplexity
	P = np.exp(-D.copy() * beta);
	sumP = sum(P);
	# print(sumP)
	H = np.log(sumP) + beta * np.sum(D * P) / sumP;
	# print("H", H)
	P = P / sumP;
	return H, P;


def x2p(X = np.array([]), tol = 1e-5, perplexity = 30.0, disPercent = 0.1):
	"""Performs a binary search to get P-values in such a way that each conditional Gaussian has the same perplexity."""

	# Initialize some variables
	print ("Computing pairwise distances...")
	(n, d) = X.shape;
	sum_X = np.sum(np.square(X), 1);
	print("SUMX ", sum_X)
	#====================================================================================modified by Ph0en1x========
	D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X) #欧氏距离矩阵
	D[D<0] = 0
	D[range(n), range(n)] = 0.000
	sD = np.sqrt(D)
	cpsD = np.sort(sD)
	for i in range(n):
		sum = 0
		tot = cpsD[i].sum()
		j = 0
		while j < n:
			sum += cpsD[i,j]
			if sum / tot > disPercent:
				break
			j += 1
		j -= 1
		j = max(j, 0)
		# print(j)
		sD[i][sD[i] > cpsD[i, j]] = np.inf
	print(sD)
	# for k in range(n):
	# 	if k % 500 == 0: print(k)
	# 	sD = minimum(sD, sD[newaxis, k, :] + sD[:, k, newaxis])
	print('floyd...')
	sD = scipy.sparse.csgraph.floyd_warshall(sD.copy()) # floyd
	mx = sD[sD < np.inf].max()
	print("max: ", mx)
	sD[sD > mx] = mx*2
	mx = sD.max()
	print("max: ", mx)
	# print("sd:", sD)
	D = np.square(sD)
	# D = D / D.max()

	# print(D)
	#=======================================================================================================
	P = np.zeros((n, n));
	beta = np.ones((n, 1));
	logU = np.log(perplexity);

	# Loop over all datapoints
	for i in range(n):

		# Print progress
		if i % 500 == 0:
			print ("Computing P-values for point ", i, " of ", n, "...")

		# Compute the Gaussian kernel and entropy for the current precision
		BETAMIN = betamin = -np.inf;
		BETAMAX = betamax =  np.inf;
		Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]; # Array of Dist(i, j) i!=j
		(H, thisP) = Hbeta(Di, beta[i]);

		# Evaluate whether the perplexity is within tolerance
		Hdiff = H - logU;
		tries = 0;
		#二分找sigma
		while np.abs(Hdiff) > tol and tries < 50:

			# If not, increase or decrease precision
			if Hdiff > 0:
				betamin = beta[i].copy();
				if betamax == np.inf or betamax == -np.inf:
					beta[i] = beta[i] * 2;
				else:
					beta[i] = (beta[i] + betamax) / 2;
			else:
				betamax = beta[i].copy();
				if betamin == np.inf or betamin == -np.inf:
					beta[i] = beta[i] / 2;
				else:
					beta[i] = (beta[i] + betamin) / 2;

			# Recompute the values
			(H, thisP) = Hbeta(Di, beta[i]);
			Hdiff = H - logU;
			tries = tries + 1;

		# Set the final row of P
		# print("iterator: ", tries)
		# print("iter %s beta is %s"%(i, beta[i]))
		P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP;
		#Pii = 0

	# Return final P-matrix
	print ("Mean value of sigma: ", np.mean(np.sqrt(1 / beta)));
	return P;


def pca(X = np.array([]), no_dims = 50):
	"""Runs PCA on the NxD array X in order to reduce its dimensionality to no_dims dimensions."""

	print ("Preprocessing the data using PCA...")
	(n, d) = X.shape;
	X = X - np.tile(np.mean(X, 0), (n, 1)); #centerization
	(l, M) = np.linalg.eig(np.dot(X.T, X)); #eigenvalue eigenvector of X^T*X
	Y = np.dot(X, M[:,0:no_dims]);
	return Y;


def tsne(X = np.array([]), no_dims = 2, initial_dims = 50, perplexity = 30.0, use_pca = True, disPercent = 0.1):
	"""Runs t-SNE on the dataset in the NxD array X to reduce its dimensionality to no_dims dimensions.
	The syntaxis of the function is Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array."""

	# Check inputs
	if isinstance(no_dims, float):
		print ("Error: array X should have type float.");
		return -1;
	if round(no_dims) != no_dims:
		print ("Error: number of dimensions should be an integer.");
		return -1;

	# Initialize variables
	#=========================================Ph0en1x======================
	Y = pca(X, no_dims).real
	#=========================kpca==
	# sigma = 2
	# XT = X.T
	# k = kpca.Polynomial_Kernel(sigma)
	# # k = kpca.RBF_Kernel(sigma)
	# p = kpca.kPCA(k, 2)
	# Y = p.fit_transform(XT)
	# Y = Y.T
	#=============================

	Y[:, 0] = 2 * (Y[:, 0] - (Y[:, 0].max() + Y[:, 0].min())/2) / (Y[:, 0].max() - Y[:, 0].min());
	Y[:, 1] = 2 * (Y[:, 1] - (Y[:, 1].max() + Y[:, 1].min())/2) / (Y[:, 1].max() - Y[:, 1].min());
	Plot.figure(0)
	Plot.title("pca")
	Plot.scatter(Y[:, 0], Y[:, 1], 20, labels, marker='.');
	Plot.savefig("pca.png")

	#===================================================================

	if use_pca:
		X = pca(X, initial_dims).real;# reduce from 784 to 50, 取实部
	(n, d) = X.shape;
	max_iter = 1000;
	initial_momentum = 0.5;
	final_momentum = 0.8;
	eta = 500;
	min_gain = 0.01;
	# Y = np.random.randn(n, no_dims);#generate the init 2d result randomly
	dY = np.zeros((n, no_dims));
	iY = np.zeros((n, no_dims));
	gains = np.ones((n, no_dims));

	# Compute P-values
	P = x2p(X, 1e-5, perplexity, disPercent=disPercent);
	#用Pij+Pji解决对称性
	P = P + np.transpose(P);
	P = P / np.sum(P);
	P = P * 4;									# early exaggeration
	P = np.maximum(P, 1e-12);

	cnt = 1
	nexiter = 0
	# Run iterations
	for iter in range(max_iter):

		# Compute pairwise affinities
		sum_Y = np.sum(np.square(Y), 1);
		num = 1 / (1 + np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T, sum_Y));
		#主对角元变为0
		num[range(n), range(n)] = 0;
		Q = num / np.sum(num);
		Q = np.maximum(Q, 1e-12);

		# Compute gradient
		PQ = P - Q;
		for i in range(n):
			dY[i,:] = np.sum(np.tile(PQ[:,i] * num[:,i], (no_dims, 1)).T * (Y[i,:] - Y), 0);

		# Perform the update
		if iter < 20:
			momentum = initial_momentum
		else:
			momentum = final_momentum
		gains = (gains + 0.2) * ((dY > 0) != (iY > 0)) + (gains * 0.8) * ((dY > 0) == (iY > 0));
		gains[gains < min_gain] = min_gain;
		iY = momentum * iY - eta * (gains * dY);
		Y = Y + iY;
		Y = Y - np.tile(np.mean(Y, 0), (n, 1));

		# Compute current value of cost function
		if (iter + 1) % 10 == 0:
			C = np.sum(P * np.log(P / Q));
			print ("Iteration ", (iter + 1), ": error is ", C)

		# Stop lying about P-values
		if iter == 100:
			P = P / 4;

		#show
		if iter == nexiter:
			Plot.figure(cnt)
			Plot.title("iter-%s"%iter)
			Plot.axis([-80,80,-80,80])
			Plot.scatter(Y[:, 0], Y[:, 1], 20, labels, marker='.');
			Plot.savefig("iter-%s.png"%iter)
			nexiter += 10 * cnt
			cnt += 1
			# Plot.show();

	# Return solution
	return Y;


if __name__ == "__main__":
	print ("Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")
	print ("Running example on 2,500 MNIST digits...")
	X = np.loadtxt("swiss_X.txt");
	labels = np.loadtxt("swiss_labels.txt");
	Y = tsne(X, 2, 50, 100.0, use_pca=False, disPercent=0.005);# use pca to reduce the dimension from 784 to 50. Then use t-SNE from 50 to 2
	Plot.figure(20)
	Plot.title("iter-1000")
	Plot.scatter(Y[:, 0], Y[:, 1], 20, labels, marker='.');
	Plot.savefig("iter-1000.png")