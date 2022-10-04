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