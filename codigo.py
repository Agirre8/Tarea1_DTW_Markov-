!pip install tslearn

import tslearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from tslearn import metrics
from tslearn.preprocessing import TimeSeriesScalerMinMax
from tslearn.piecewise import PiecewiseAggregateApproximation
import warnings
warnings.filterwarnings('ignore')

stocks_df = pd.read_csv("../input/moroccan-stock-prices/stocks.csv")

stocks_df.date = pd.to_datetime(stocks_df.date, format='%d/%m/%Y')

(stocks_df.isnull().sum()*100 / stocks_df.shape[0]).sort_values(ascending = False).head(15)

stocks_df.drop(columns = ["SAMIR", "Diac Salaf", "Aradei Capital", "Mutandis", "Immr Invest"], inplace = True)
(stocks_df.isnull().sum()*100 / stocks_df.shape[0]).sort_values(ascending = False).head(15)
stocks_df = stocks_df.resample('7D', on = 'date').first().reset_index(drop = True)

stocks_df.index = stocks_df.date
stocks_df.drop("date", axis = 1, inplace = True)
stocks = stocks_df

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(stocks_df.index, stocks_df["MASI"])
ax.grid()
ax.set_title('MASI Index Movements')
ax.set_ylabel('MASI')
ax.set_xlabel('date')
fig.tight_layout();

cols = stocks_df.columns

fig, axs = plt.subplots(10,7,figsize=(35,35))
for i in range(10):
    for j in range(7):
        axs[i, j].plot(stocks_df[cols[i*7+j]].values)
        axs[i, j].set_title(cols[i*7+j])
plt.show()

ts = np.array(stocks.T).reshape(stocks.T.shape[0], stocks.T.shape[1], 1)
ts = TimeSeriesScalerMinMax().fit_transform(ts)

n_segments = 10
paa = PiecewiseAggregateApproximation(n_segments = n_segments)
ts = paa.fit_transform(ts)
km = TimeSeriesKMeans(n_clusters = 4, random_state = 42, metric = 'dtw')
y_pred = km.fit_predict(ts)
s = silhouette_score(ts, y_pred, metric='dtw')
print("K-means metrics : ")
print(f"Silhouette score = {s}, \nInertia = {km.inertia_}")

clusters_df = pd.DataFrame({'Company':stocks.columns, 'Cluster': y_pred}).sort_values(by = 'Cluster')
clusters_df.reset_index(drop = True)

for yi in range(4):
    plt.subplot(3, 3, 4 + yi)
    for xx in ts[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, ts.shape[1])
    plt.ylim(-.5, 1.5)
    plt.text(0.55, 0.85,'Cluster %d' % (yi),
             transform=plt.gca().transAxes)
    plt.gcf().set_size_inches(15, 15)
    if yi == 1:
        plt.title("DBA $k$-means")

cluster_c = clusters_df["Cluster"].value_counts().sort_index()
cluster_n = [f"Cluster{str(i)}" for i in range(km.n_clusters)]
fig, ax = plt.subplots(1, 1,figsize=(15,5))
ax.set_title("Cluster Distribution for KMeans")
ax.bar(cluster_n, cluster_c)
plt.show()