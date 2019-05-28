import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets.samples_generator import make_blobs
x, y = make_blobs(n_samples = 300, centers = 5, cluster_std = 0.60)

plt.scatter(x[:, 0], x[:, 1])
plt.show()

wcss = []

from sklearn.cluster import KMeans

for i in range(1, 11):
    km_clus = KMeans(n_clusters = i)
    km_clus.fit(x)
    wcss.append(km_clus.inertia_)
    
plt.plot(range(1, 11), wcss)
plt.show()

km = KMeans(n_clusters = 5)
km.fit(x)
y_cat = km.predict(x)

plt.scatter(x[y_cat == 0, 0], x[y_cat == 0, 1], c = "r", label = "0")
plt.scatter(x[y_cat == 1, 0], x[y_cat == 1, 1], c = "g", label = "1")
plt.scatter(x[y_cat == 2, 0], x[y_cat == 2, 1], c = "b", label = "2")
plt.scatter(x[y_cat == 3, 0], x[y_cat == 3, 1], c = "cyan", label = "3")
plt.scatter(x[y_cat == 4, 0], x[y_cat == 4, 1], c = "y", label = "4")
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import make_blobs
x, y = make_blobs(n_samples = 300, centers = 5, cluster_std = 0.60)

plt.scatter(x[:, 0], x[:, 1])
plt.show()

wcv = []

from sklearn.cluster import KMeans

for i in range(1, 11):
    km = KMeans(n_clusters = i)
    km.fit(x, y)
    wcv.append(km.inertia_)

plt.plot(range(1, 11), wcv)
plt.show()

km = KMeans(n_clusters = 5)
km.fit(x, y)

y_pred = km.predict(x)

plt.scatter(x[y_pred == 0, 0], x[y_pred == 0, 1])
plt.scatter(x[y_pred == 1, 0], x[y_pred == 1, 1])
plt.scatter(x[y_pred == 2, 0], x[y_pred == 2, 1])
plt.scatter(x[y_pred == 3, 0], x[y_pred == 3, 1])
plt.scatter(x[y_pred == 4, 0], x[y_pred == 4, 1])
plt.show()

































