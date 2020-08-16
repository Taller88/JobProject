import pandas as pd

first_highschool=pd.read_csv('first_h_highschool_st.csv')
first_vocationhighschool=pd.read_csv('first_h_vocationalhighschool_st.csv')
fourth_m_highschool=pd.read_csv('f4_m_highschool_st.csv')
fourth_m_earlyGradu=pd.read_csv('f4_m_earlygrad_st.csv')
fourth_m_vocation=pd.read_csv('f4_m_vocationhighschool_st.csv')

print(fourth_m_vocation.shape)
print(fourth_m_highschool.shape)
# print(fourth_m_earlyGradu.shape)
h_high=pd.concat([first_highschool,first_vocationhighschool])
m_high=pd.concat([fourth_m_highschool, fourth_m_vocation])

h_names=h_high.columns.tolist()
m_names=m_high.columns.tolist()
a=[]
for i in m_names:
    if i in h_names:
        a.append(i)
print(len(a))
print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
fourth=pd.concat([fourth_m_highschool, fourth_m_earlyGradu,fourth_m_vocation])
fourth_names=fourth_m_earlyGradu.columns.tolist()
print(fourth.shape)
third_highschool=pd.concat([first_highschool,fourth_m_earlyGradu,fourth_m_highschool,fourth_m_vocation])
print(third_highschool)