import pandas as pd

df=pd.read_csv('J007C.csv')
print(df.iloc[:,:-1])

data=df[::-1]
data.to_csv('real_J007C.csv', index=False)
