import pandas as pd
from sklearn.decomposition import PCA

df=pd.read_csv('total_reduce_resultVariable_third.csv')

grade_pcaData=df[['BYS11001','BYS11002','BYS11003','BYS11004','BYS11005','BYS11006']]
pca=PCA(n_components=1)
X=pca.fit_transform(grade_pcaData)

print(grade_pcaData)
print(X)