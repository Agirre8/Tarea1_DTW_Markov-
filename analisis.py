import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

x=pd.read_csv('stocks.csv')
list=[x['date'],x['Addoha'],x['AFMA'],x['Afric Indus']]
y=pd.DataFrame(list)
dataset_bueno=y.transpose()
dataset_bueno=dataset_bueno.dropna()
print(dataset_bueno)
for column in dataset_bueno.columns:
    if column!='date':
        plt.plot(dataset_bueno[column])
plt.show()
plt.title('Estados')


