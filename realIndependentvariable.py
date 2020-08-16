import pandas as pd

data=pd.read_csv('RealTotal.csv')
names=data.columns.tolist()
count_minus=data.iloc[0,:]


checked_names=[]
test=len(count_minus)
for i in range(test):
    if count_minus[i] ==0:
        checked_names.append(names[i])
# print(checked_names)
# print(len(checked_names))
# print(data)
df=pd.DataFrame()

list = [None] * len(checked_names)
for number, col in enumerate(checked_names):
    list[number] = data[col]
for i in range(len(checked_names)):
    print(list[i])
    print("-----------------")
for i, j in enumerate(checked_names):
    df[j]=list[i]

print(df)

df.to_csv("realTotal2.csv",index=False)
