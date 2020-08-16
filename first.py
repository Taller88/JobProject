import pandas as pd
first_h3_student=pd.read_csv('first_h_highschool_st.CSV')
first_h3_teacher=pd.read_csv('first_h_teacher.csv')
first_h3_household=pd.read_csv('first_h_household.csv')
first_h3_advisor=pd.read_csv('first_h_highschool_ad.csv')
first_h3_vostudent=pd.read_csv('first_h_vocationalhighschool_st.csv')
first_h3_voAd=pd.read_csv('first_h_vocationalhighschool_ad.csv')
print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")

student_names=first_h3_student.columns.tolist()
voAd_names=first_h3_voAd.columns.tolist()
stAd_names=first_h3_advisor.columns.tolist()

print(student_names)
print(len(first_h3_student))
print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")
print(first_h3_student.shape)
print(first_h3_advisor.shape)
result=pd.merge(first_h3_student, first_h3_advisor, on=['BYAID'], suffixes=("", "_yasd"))
print(result)
print(first_h3_vostudent.shape)
print(first_h3_voAd.shape)
result2=pd.merge(first_h3_vostudent, first_h3_voAd, on=['BYAID'], suffixes=("", "_yasd"))
print(result2)
result3=pd.concat([result, result2])
print(result3)
print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")
# for i in name3:
#     if i not in name2:
#         data.append(i)
# print(data)
# print(len(data))
result4=pd.merge(result3, first_h3_household, on = ['BYHID'], suffixes=("", "_yasd"), how='left')
result5=pd.merge(result4, first_h3_teacher, on=['BYTID'], suffixes=("", "_yasd"), how='left')

print(result5)
first_names=result5.columns.tolist()
dropData=[]
for i in first_names:
    if '_yasd' in i :
        dropData.append(i)
#result5.to_csv("jinwo.csv", index=False)
result5.drop(dropData, axis='columns',inplace=True)
print(result5)
# if 'a' in te:
#     print('thank')
# else:
#     print('sorry')
#result5.to_csv("jinwo.csv", index=False)
print(first_h3_student.shape)
print(first_h3_vostudent.shape)
print(first_h3_household.shape)
print(first_h3_teacher.shape)
