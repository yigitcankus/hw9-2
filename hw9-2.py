import timeit

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn import datasets

# Loading the data from Sklearn's datasets
# iris = datasets.load_iris()
# X = iris.data
# y = iris.target
#
# # Standarizing the features
# scaler = StandardScaler()
# X_std = scaler.fit_transform(X)
#
# # Defining the k-means
# kmeans_küme = KMeans(n_clusters=3, random_state=124)
#
# # Fit model
# kmeans_küme.fit(X_std)
# y_pred = kmeans_küme.predict(X_std)
#
# plt.scatter(X_std[:, 0], X_std[:, 1], c=y_pred, s=50, cmap='viridis')
# plt.show()
#
#
# minikmeans_cluster = MiniBatchKMeans(
#     n_clusters=3,
#     batch_size=50)
#
# minikmeans_cluster.fit(X_std)
# y_pred_mini_batch = minikmeans_cluster.predict(X_std)
# plt.scatter(X_std[:, 0], X_std[:, 1], c=y_pred, s=50, cmap='viridis')
# plt.show()


# Daha güzel bir görselleştirme yaptım.
# Cluster sayısını 4 yaptıktan sonra 0 ile gösterdiği kırmızı renkteki parçaları ayırmaya başladı. Onları da değiştirmeye çalıştı.
# Arttırmaya devam ettikçe daha önceden grupladığı kısımları parçlamaya başladı.,



#####################################################################################################################
#####################################################################################################################
#####################################################################################################################

#Heart disease

heartdisease_df = pd.read_csv("cleveland-0_vs_4.csv")

heartdisease_df = heartdisease_df.replace(to_replace='negative', value=0)
heartdisease_df = heartdisease_df.replace(to_replace='positive', value=1)

heartdisease_df["ca"] = heartdisease_df.ca.replace({'<null>':0})
heartdisease_df["ca"] = heartdisease_df["ca"].astype(np.int64)

heartdisease_df["thal"] = heartdisease_df.thal.replace({'<null>':0})
heartdisease_df["thal"] = heartdisease_df["thal"].astype(np.int64)


X = heartdisease_df.iloc[:, :13]
y = heartdisease_df.iloc[:, 13]



scaler = StandardScaler()
X_std = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=2)
kmeans.fit(X_std)
y_kmeans = kmeans.predict(X_std)

plt.scatter(X_std[:, 0], X_std[:, 1], c=y_kmeans, s=50, cmap='viridis')

plt.show()

#modelimiz kişileri sağ ve sol olarak bölüyor ama hangisinin hangisi olduğunu anlayamadım.









