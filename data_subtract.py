import pandas as pd

df=pd.read_csv('total_except_null_second.csv')
df2=pd.read_csv('Real_Final.csv')

names=df.columns.tolist()
names2=df2.columns.tolist()

checked_names=[]
for i in names:
    if i not in names2:
        checked_names.append(i)
total=df[['BYS02001', 'BYS02002', 'BYS02005', 'BYS02006', 'BYS02007', 'BYS02008', 'BYS02009', 'BYS02010', 'BYS02011', 'BYS03002', 'BYS03003', 'BYS03004', 'BYS04001', 'BYS04002', 'BYS04003', 'BYS04004', 'BYS04005', 'BYS04006', 'BYS07015', 'BYS07016', 'BYS07017', 'BYS07018', 'BYS07019', 'BYS07020', 'BYS07021', 'BYS07028', 'BYS08008', 'BYS10004', 'BYS11007', 'BYS11008', 'BYS11009', 'BYS11010', 'BYS11011', 'BYS11012', 'BYS11013', 'BYS11014', 'BYS11015', 'BYS11016', 'BYS12025', 'BYS20022', 'BYS22004', 'BYS22005', 'BYS22007', 'BYS22008', 'BYS22009', 'BYS22016', 'BYS23001', 'BYS24003', 'BYS24004', 'BYS26008', 'BYS32001', 'BYS32002', 'BYS32003', 'BYS32004', 'BYS32005', 'BYS32006', 'BYS32007', 'BYS32008', 'BYS32009', 'BYS32010']]
total['BYSID']=df['BYSID']

total.to_csv('selected_data.csv', index=False)