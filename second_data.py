import pandas as pd
h_college=pd.read_csv('f2_h_college_student.csv')
h_youth = pd.read_csv('f2_h_youth.csv')
m_youth = pd.read_csv('f2_m_youth.csv')
m_high = pd.read_csv('f2_m_highschool_st.csv')
m_vo = pd.read_csv('f2_m_vocationalhighschool_st.csv')
nomal_highschool=pd.read_csv('first_h_highschool_st.csv')
vo_highschool = pd.read_csv('first_h_vocationalhighschool_st.csv')
first_middleSchool=pd.read_csv('first_m_middleschool_st.csv')
bysid_job = pd.read_csv('bysid_j007_2.csv')

result=pd.concat([nomal_highschool, vo_highschool])
result=pd.concat([result,first_middleSchool])
print(result)

total=pd.merge(result,bysid_job, on="BYSID", how="inner" )
print(total.shape)
print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')


print(h_college.shape)
print(h_youth.shape)
high=pd.concat([h_college, h_youth])
print(len(high))
print(high.shape)


print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')


print(m_high.shape)
print(m_vo.shape)

middle=pd.concat([m_high, m_vo])
print(middle.shape)


print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')

print(h_college.shape)
print(middle.shape)
totalTest=pd.concat([middle,h_college])
print(totalTest.shape)
print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
print(total.shape)
print(middle.shape)
student=pd.concat([total, middle])
print(student.shape)