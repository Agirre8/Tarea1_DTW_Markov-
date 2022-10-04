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