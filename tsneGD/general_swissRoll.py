import sklearn
import sklearn.datasets
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

num_of_point = 2500
num_of_class = 4

X, t = sklearn.datasets.make_swiss_roll(num_of_point, 0.0)
f1 = open("out.txt", "w")
f2 = open("tar.txt", "w")
mx = max(t)
mi = min(t)
print(mx, mi)
step = (mx - mi) // num_of_class
t = (t-mi)//step


for i in range(num_of_point):
	print("%f %f %f" %(X[i,0], X[i,1], X[i,2]), file = f1)
	print(t[i], file = f2)
	
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], X[:,2], c=t, marker='.')
plt.show()