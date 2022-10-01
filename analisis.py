import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np

x=pd.read_csv('stocks.csv')
list=[x['date'],x['Addoha'],x['AFMA'],x['Afric Indus']]
y=pd.DataFrame(list)
dataset_bueno=y.transpose()
dataset_bueno=dataset_bueno.dropna()
dataset_bueno=dataset_bueno.drop(['date'],axis=1)
print(dataset_bueno)
for column in dataset_bueno.columns:

    plt.plot(dataset_bueno[column])
plt.show()
plt.title('Estados')




