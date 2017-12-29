import kpca
import numpy as np
import pylab as Plot
from GD_tsne import pca

sigma = 10
X = np.loadtxt("swiss_X.txt");
labels = np.loadtxt("swiss_labels.txt");
XT = X.T
# k = kpca.Polynomial_Kernel(sigma)
k= kpca.RBF_Kernel(sigma)
p = kpca.kPCA(k, 2)
Y = p.fit_transform(XT)
Y = Y.T

print(X.shape)
print(Y.shape)
Plot.figure(20)
Plot.title("kpca")
Plot.scatter(Y[:, 0], Y[:, 1], 20, labels, marker='.');
Plot.show()