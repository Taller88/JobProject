import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score


poly=PolynomialFeatures(degree=2, interaction_only=True)
lr_clf=LogisticRegression(solver='lbfgs', multi_class='multinomial', C=1)
svm=SVC(gamma='auto', C=1.0)
df=pd.read_csv('Real_Final.csv')
categorical_names=['father_job','hope_job_reason','club','hope_job','AT_05','certification','eduColl_Ma','eduColl_Fa','h_major','col_major','sch_code','BYS04008', 'BYS04009', 'BYS07002', 'BYS07006', 'BYS07014', 'BYS08001', 'BYS08002', 'BYS08005', 'BYS09002', 'BYS09003', 'BYS22001','BYS22002','BYS23002', 'BYS23003', 'BYS23004', 'BYS24002', 'BYS16001', 'BYS20021', 'BYS27020','BYS08007']
numerical_names=['today_1','today_2','today_3','today_4','today_5','today_6','today_7','today_8','today_9','today_10','today_11','today_12','today_13','today_14','today_15','today_16','today_17','today_18','today_19','library','BYS25002','AT_32','AT_33','AT_34','eduYear_Ma','eduYear_Fa', 'F8Y140_Influence', 'F8Y140_Feeling', 'F8Y140_consideration', 'F8Y140_Intuition', 'F8Y140_Sensing','BYS02003', 'BYS02004', 'BYS02012', 'BYS04007', 'BYS07001', 'BYS07025', 'BYS07026', 'BYS07027', 'BYS07029', 'BYS08003', 'BYS08004', 'BYS09001', 'BYS09004', 'BYS10001', 'BYS10002', 'BYS10003', 'BYS10005', 'BYS11001', 'BYS11002', 'BYS11003', 'BYS11004', 'BYS11005', 'BYS11006', 'BYS13001', 'BYS22003', 'BYS22006', 'BYS22012', 'BYS22013', 'BYS22014', 'BYS22015', 'BYS22017', 'BYS25001', 'BYS25003']
discrete_names=['GENDER', 'BYS03001', 'BYS04010', 'BYS05001', 'BYS05003', 'BYS05006', 'BYS06001', 'BYS06004', 'BYS07003', 'BYS07022', 'BYS12001', 'BYS12004', 'BYS12007', 'BYS12010', 'BYS12013', 'BYS12016', 'BYS12019', 'BYS12022', 'BYS14001', 'BYS14013', 'BYS14025', 'BYS15001', 'BYS20001', 'BYS20007', 'BYS20009', 'BYS20011', 'BYS20013', 'BYS20015', 'BYS20017', 'BYS20018', 'BYS20019', 'BYS20020', 'BYS21001', 'BYS22010', 'BYS24001', 'BYS24005', 'BYS26001', 'BYS26002', 'BYS26003', 'BYS26004', 'BYS26005', 'BYS26006', 'BYS26007', 'BYS26009', 'BYS27001', 'BYS27003', 'BYS27006', 'BYS27012']

except_data=['BYSID','BYHID', 'BYAID', 'BYTID', 'BYSCLASS' ]


categorical_data=df[categorical_names].astype('str')
numerical_data=df[numerical_names]
numerical_data['BYSID']=df['BYSID']

discrete_data=df[discrete_names]
discrete_data['BYSID']=df['BYSID']
categorical_data_oneHot=pd.get_dummies(categorical_data)
categorical_data_oneHot['BYSID']=df['BYSID']
print(categorical_data_oneHot)
data=pd.merge(categorical_data_oneHot, numerical_data, on='BYSID')
data=pd.merge(data,discrete_data, on='BYSID')
del data['BYSID']
# -----------------------------------------------변수 선택-----------------------------------------------------
# 애매한 변수: 'BYS22010(믿어주는 사람 유무)','BYS22003(자기자신이 공부를 잘한다)','BYS21001(아르바이트 경험 유무)''BYS24002(종교)','BYS26003'(가정의 경제형편의 걱정 정도),'BYS16001(어느수준까지 교육)','BYS20017'(어떻게 살것인지 대화),'BYS23002'(직업선택할 때 중요한 요소)
# 감소한 변수: 'BYS09003(용돈어디에 사용)','BYS08005(평일에 뭐하는지)''BYS07022(형제자매 유무)''BYS07029(가족끼리 운동)','BYS05006(봉사활동 경험)''BYS04008~9(친구들 대화 유형)','BYS22014(무엇이 중요한지 아는지 선호도)','BYS22006(미래에 대한 꿈이 있다)','BYS22001(인생살면서 가장 중효한 것)',
#mustHave_columns=['h_major','F8Y140_Influence',  ,'today_1','today_2','today_3','today_4','today_5','today_8','today_9','today_11','today_12','today_13','today_14','today_15','today_16','today_17','today_18','today_19']
# 'father_job','BYS12013','BYS12010','BYS11004','BYS09002','BYS25002','BYS21001','sch_code','h_major','col_major','GENDER','F8Y140_Influence', 'F8Y140_Feeling', 'F8Y140_consideration', 'F8Y140_Intuition', 'F8Y140_Sensing', 'AT_05', 'AT_32', 'AT_33', 'AT_34', 'eduYear_Ma', 'eduYear_Fa', 'eduColl_Ma', 'eduColl_Fa'
mustHave_columns=['col_major','today_9']
need_oneHot=[]
numeric=[]
discrete=[]
for i in mustHave_columns:
    if i in categorical_names:
        need_oneHot.append(i)
    elif i in numerical_names:
        numeric.append(i)
    else:
        discrete.append(i)

if need_oneHot:

    selected_oneHot=pd.get_dummies(categorical_data[need_oneHot])
    selected_oneHot['BYSID']=df['BYSID']
    selected_numeric = numerical_data[numeric]
    selected_numeric['BYSID']=df['BYSID']
    selected_discrete=discrete_data[discrete]
    selected_discrete['BYSID']=df['BYSID']
    selected_data=pd.merge(selected_numeric, selected_oneHot, on='BYSID')
    selected_data=pd.merge(selected_data, selected_discrete, on='BYSID')
else:
    selected_numeric=numerical_data[numeric]
    selected_numeric['BYSID']=df['BYSID']
    selected_discrete=discrete_data[discrete]
    selected_discrete['BYSID']=df['BYSID']
    selected_data=selected_numeric
# selected_data: 추출한 독립변수
del selected_data['BYSID']
data=selected_data
print(data)

#
# -----------------------------------------------변수 선택-----------------------------------------------------



target=df['J007C']
X_train, X_test, y_train,y_test=train_test_split(data,target, random_state=7, test_size=0.3)
lr_clf.fit(X_train, y_train)
print("Logistic Train Score: ", lr_clf.score(X_train, y_train))
print("Logistic Test Score: ", lr_clf.score(X_test, y_test))

svm.fit(X_train, y_train)
print("SVM Score: ", svm.score(X_train,y_train))
print("SVM Score: ", svm.score(X_test, y_test))
X_train_poly=poly.fit_transform(X_train)
X_test_poly=poly.transform(X_test)

lr_clf.fit(X_train_poly, y_train)
print("Logistic Train Score: ", lr_clf.score(X_train_poly, y_train))
print("Logistic Test Score: ", lr_clf.score(X_test_poly, y_test))

svm.fit(X_train_poly, y_train)
print("SVM Score: ", svm.score(X_train_poly,y_train))
print("SVM Score: ", svm.score(X_test_poly, y_test))

# scores=cross_val_score(lr_clf,data,target, cv=10)
# print(scores)


print(X_train)