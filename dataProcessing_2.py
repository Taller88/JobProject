import pandas as pd

middle=pd.read_csv('first_m_middleschool_st.csv')
high_vo=pd.read_csv('first_h_vocationalhighschool_st.csv')
high=pd.read_csv('first_h_highschool_st.csv')


middle_names=middle.columns.tolist()
high_vo_names=high_vo.columns.tolist()
high_names=high.columns.tolist()
print("middle Shape: ", middle.shape)
print("High_vocation Shape: ", high_vo.shape)
print("High Shape: ", high.shape)


total_names=[]
for i in high_names:
    if i in high_vo_names:
        total_names.append(i)
print(len(total_names))
total_names2=[]
for i in middle_names:
    if i in total_names:
        total_names2.append(i)
print(len(total_names))