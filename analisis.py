import pandas as pd
x=pd.read_csv('stocks.csv')
list=[x['Addoha'],x['AFMA'],x['Afric Indus']]
y=pd.DataFrame(list)
dataset_bueno=y.transpose()
print(dataset_bueno)