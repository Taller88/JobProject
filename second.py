import pandas as pd
second_h3_student = pd.read_csv('f2_h_college_student.CSV')
second_h3_household = pd.read_csv('f2_h_household.csv')
second_h3_youth = pd.read_csv('f2_h_youth.csv')
print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")

student_names=second_h3_student.columns.tolist()
house_names=second_h3_household.columns.tolist()
youth_names=second_h3_youth.columns.tolist()
print(second_h3_student.shape)
print(second_h3_household.shape)
for i in student_names:
    if i in house_names:
        print(i)
result=pd.merge(second_h3_student, second_h3_household, on=['BYHID'], suffixes=("", "_yasd"))
print(result.shape)
print(second_h3_student['BYHID'])
# print(result)
# print(first_h3_vostudent.shape)
# print(first_h3_voAd.shape)
# result2=pd.merge(first_h3_vostudent, first_h3_voAd, on=['BYAID'], suffixes=("", "_yasd"))
# print(result2)
# result3=pd.concat([result, result2])
# print(result3)
print("ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ")
# for i in name3:
#     if i not in name2:
#         data.append(i)
# print(data)
# print(len(data))
# result4=pd.merge(result3, first_h3_household, on = ['BYHID'], suffixes=("", "_yasd"), how='left')
# result5=pd.merge(result4, first_h3_teacher, on=['BYTID'], suffixes=("", "_yasd"), how='left')
#
# print(result5)
# first_names=result5.columns.tolist()
# dropData=[]
# for i in first_names:
#     if '_yasd' in i :
#         dropData.append(i)
# result5.drop(dropData, axis='columns',inplace=True)
# print(result5)
# if 'a' in te:
#     print('thank')
# else:
#     print('sorry')
#result5.to_csv("jinwo.csv", index=False)
# print(first_h3_student.shape)
# print(first_h3_vostudent.shape)
# print(first_h3_household.shape)
# print(first_h3_teacher.shape)
