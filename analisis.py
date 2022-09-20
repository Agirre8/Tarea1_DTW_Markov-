import pandas as pd
x=pd.read_csv('stocks.csv')
list=[x['Addoha'],x['AFMA'],x['Afric Indus']]
y=pd.DataFrame(list)
dataset_bueno=y.transpose()
dataset_bueno=dataset_bueno.dropna()
print(dataset_bueno)