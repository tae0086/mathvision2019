import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import mahalanobis
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA

# read apple A and B data respectively
column_names = ['당도', '밀도', '색상', '수분함량']
data_a = pd.read_csv('./data/data_a.txt', header=None, names=column_names)
data_b = pd.read_csv('./data/data_b.txt', header=None, names=column_names)

# PCA
pca = PCA(n_components=2)
data_low = pca.fit_transform(pd.concat([data_a, data_b], axis=0))
# print(pca.components_)
data_low_a = data_low[:1000]
data_low_b = data_low[1000:]

# split column into x, y coordinates
x_a = data_low_a[:, 0]
y_a = data_low_a[:, 1]
x_b = data_low_b[:, 0]
y_b = data_low_b[:, 1]

# scatter plot
plt.plot(x_a, y_a, 'ro', label='apple A')
plt.plot(x_b, y_b, 'bo', label='apple B')
plt.legend()
plt.show()

# gaussian distribution for 3D gaussian plot
# in apple A data
mean_a = np.mean(data_low_a, axis=0)
cov_a = np.cov(data_low_a.T)
X_a, Y_a = np.meshgrid(x_a, y_a)
pos_a = np.dstack((X_a, Y_a))
pdf_a = multivariate_normal.pdf(x=pos_a, mean=mean_a, cov=cov_a)
# in apple B data
mean_b = np.mean(data_low_b, axis=0)
cov_b = np.cov(data_low_b.T)
X_b, Y_b = np.meshgrid(x_b, y_b)
pos_b = np.dstack((X_b, Y_b))
pdf_b = multivariate_normal.pdf(x=pos_b, mean=mean_b, cov=cov_b)

# plot apple A gaussian
figure_a = plt.figure()
axes_a = plt.axes(projection='3d')
axes_a.plot_surface(X_a, Y_a, pdf_a, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
# plot apple B gaussian
figure_b = plt.figure()
axes_b = plt.axes(projection='3d')
axes_b.plot_surface(X_b, Y_b, pdf_b, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

plt.show()

# same process for TEST data
data_test = pd.read_csv('./data/test.txt', header=None, names=column_names)
data_test_low = np.matmul(data_test.values, pca.components_.T)
test_sample_1 = data_test_low[0]
test_sample_2 = data_test_low[1]

# Mahalonobis distance
# in apple A
print(mahalanobis(u=test_sample_1, v=mean_a, VI=np.linalg.inv(cov_a)))  # 0.8295541478289046
print(mahalanobis(u=test_sample_2, v=mean_a, VI=np.linalg.inv(cov_a)))  # 8.112025001526678
# in apple B
print(mahalanobis(u=test_sample_1, v=mean_b, VI=np.linalg.inv(cov_b)))  # 4.771411327871158
print(mahalanobis(u=test_sample_2, v=mean_b, VI=np.linalg.inv(cov_b)))  # 0.9875768904541256

# scatter plot with test data
plt.plot(x_a, y_a, 'ro', label='apple A')
plt.plot(x_b, y_b, 'bo', label='apple B')
plt.plot(test_sample_1[0], test_sample_1[1], 'g*', markersize=12, label='test 1')
plt.plot(test_sample_2[0], test_sample_2[1], 'gh', markersize=12, label='test 2')
plt.legend()
plt.show()
