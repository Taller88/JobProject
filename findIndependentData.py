import pandas as pd

total=pd.read_csv('total.csv')
test=total
test2=total.dropna(axis=1)
a=[]
for i in test.columns.tolist():
    if i  not in test2.columns.tolist():
        a.append(i)
print(a)

print(test2.shape)
print(test.shape)

test2.to_csv("RealTotal.csv",index=False)

