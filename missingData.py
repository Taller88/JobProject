import pandas as pd


vost=pd.read_csv('first_h_vocationalhighschool_st.csv')
st= pd.read_csv('first_h_highschool_st.csv')

vost_names=vost.columns.tolist()
st_names=st.columns.tolist()

check=[]
for i in st_names:
    if i in vost_names:
       check.append(i)
print(len(check))
print(len(vost_names))
print(len(st_names))
print(st['GENDER'])
