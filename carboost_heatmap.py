import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
df=pd.read_csv('xg_heatmap.csv')
data=df.values

sns.heatmap(data , cmap='YlGnBu')



plt.savefig('xg_Heatmap.jpg')

plt.show()


df2=pd.read_csv('rf_heatmap.csv')
data2=df.values

sns.heatmap(data2, cmap='YlGnBu')

plt.savefig('rf_Heatmap.jpg')

plt.show()