from sklearn.cluster import KMeans

import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
import seaborn as sb

print(sb.__version__)



df = pd.read_csv('./datasets/var.csv')
print(df)


sb.pairplot(df)
X = np.array(df.drop('target',axis=1))
print(X)


kmeans = KMeans(n_clusters=3,random_state = 0,n_jobs=1)

kmeans.fit(X)

print(kmeans.labels_)