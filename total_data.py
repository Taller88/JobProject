import pandas as pd
nomal_highschool=pd.read_csv('first_h_highschool_st.csv')
vo_highschool = pd.read_csv('first_h_vocationalhighschool_st.csv')
first_middleSchool=pd.read_csv('first_m_middleschool_st.csv')
bysid_job = pd.read_csv('bysid_j007_2.csv')

result=pd.concat([nomal_highschool, vo_highschool])
result=pd.concat([result,first_middleSchool])
print(result)

total=pd.merge(result,bysid_job, on="BYSID", how="inner" )
print(total.shape)

# total.to_csv("total.csv",index=False)

# stu = pd.merge(vo_highschool, nomal_highschool, on='BYSID', how='inner', suffixes=("", "_jinwoo"))
# first_total_student=pd.merge(stu, first_middleSchool, on="BYSID",  how='inner', suffixes=("", "_jinwoo"))
# stu_names=stu.columns.tolist()
# student=stu.set_index("BYSID")
# total=first_total_student.loc[find_bysid]
# total=total.sort_values(['BYSID'])
# print(total)
#
#
# names=vo_highschool.columns.tolist()
# names2=nomal_highschool.columns.tolist()
# names3=first_middleSchool.columns.tolist()
# a=[]
# for i in names:
#     for j in names2:
#         if i == j :
#             a.append(j)
# print(a)
# print(len(a))
#
#
# total.to_csv("jin.csv", index=False)
#result5.drop(dropData, axis='columns',inplace=True)