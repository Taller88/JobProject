import pandas as pd
import numpy as np
from sklearn.metrics import recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier


poly=PolynomialFeatures(degree=5, interaction_only=True)
df=pd.read_csv('total_personality.csv')

total_names=df.columns.tolist()
categorical_names=['sch_code','BYS04008', 'BYS04009', 'BYS07002', 'BYS07006', 'BYS07014', 'BYS08001', 'BYS08002', 'BYS08005', 'BYS09002', 'BYS09003', 'BYS22001','BYS22002','BYS23002', 'BYS23003', 'BYS23004', 'BYS24002', 'BYS16001', 'BYS20021', 'BYS27020','BYS08007']
numerical_names=['F8Y140_Influence', 'F8Y140_Feeling', 'F8Y140_consideration', 'F8Y140_Intuition', 'F8Y140_Sensing','BYS02003', 'BYS02004', 'BYS02012', 'BYS04007', 'BYS07001', 'BYS07025', 'BYS07026', 'BYS07027', 'BYS07029', 'BYS08003', 'BYS08004', 'BYS09001', 'BYS09004', 'BYS10001', 'BYS10002', 'BYS10003', 'BYS10005', 'BYS11001', 'BYS11002', 'BYS11003', 'BYS11004', 'BYS11005', 'BYS11006', 'BYS13001', 'BYS22003', 'BYS22006', 'BYS22012', 'BYS22013', 'BYS22014', 'BYS22015', 'BYS22017', 'BYS25001', 'BYS25003']
discrete_names=['GENDER', 'BYS03001', 'BYS04010', 'BYS05001', 'BYS05003', 'BYS05006', 'BYS06001', 'BYS06004', 'BYS07003', 'BYS07022', 'BYS12001', 'BYS12004', 'BYS12007', 'BYS12010', 'BYS12013', 'BYS12016', 'BYS12019', 'BYS12022', 'BYS14001', 'BYS14013', 'BYS14025', 'BYS15001', 'BYS20001', 'BYS20007', 'BYS20009', 'BYS20011', 'BYS20013', 'BYS20015', 'BYS20017', 'BYS20018', 'BYS20019', 'BYS20020', 'BYS21001', 'BYS22010', 'BYS24001', 'BYS24005', 'BYS26001', 'BYS26002', 'BYS26003', 'BYS26004', 'BYS26005', 'BYS26006', 'BYS26007', 'BYS26009', 'BYS27001', 'BYS27003', 'BYS27006', 'BYS27012']

except_data=['BYSID','BYHID', 'BYAID', 'BYTID', 'BYSCLASS' ]
#제거이유: 기본 ID와 같은 속성들과 수상종목인데 선택된 값은 0,1로 구성되어 배제


categorical_data=df[categorical_names].astype('str')
numerical_data=df[numerical_names]
numerical_data['BYSID']=df['BYSID']

discrete_data=df[discrete_names]
discrete_data['BYSID']=df['BYSID']
categorical_data_oneHot=pd.get_dummies(categorical_data)
categorical_data_oneHot['BYSID']=df['BYSID']

# concat으로 했을경우 BYSID가 KEY로 되지 않는듯 ...
# result_concat=pd.concat([categorical_data, numerical_data])
result_merge=pd.merge(categorical_data_oneHot, numerical_data, on='BYSID')
result_merge=pd.merge(result_merge,discrete_data, on='BYSID')
print('-------------------------------------')
print(categorical_data_oneHot.shape)
print(numerical_data.shape)
print(discrete_data.shape)
print('-------------------------------------')
print(result_merge)
print('-------------------------------------')
result_merge.drop(['BYSID'], axis='columns', inplace=True)
print(result_merge)


must_include_cols = ['F8Y140_Influence', 'F8Y140_Feeling', 'F8Y140_consideration', 'F8Y140_Intuition', 'F8Y140_Sensing', 'BYS07029', 'BYS11003', 'BYS02004', 'sch_code_0', 'sch_code_1', 'sch_code_2', 'sch_code_3', 'sch_code_4', 'sch_code_5', 'sch_code_6', 'sch_code_7', 'sch_code_8', 'sch_code_9', 'sch_code_10', 'BYS27020_1', 'BYS27020_2', 'BYS27020_3', 'BYS27020_4', 'BYS27020_5', 'BYS27020_6', 'BYS27020_7', 'BYS27020_8', 'BYS27020_9', 'BYS27020_10', 'BYS27020_11', 'BYS27020_12', 'BYS27020_13', 'BYS27020_14', 'BYS27020_15', 'BYS20021_1', 'BYS20021_2', 'BYS20021_3', 'BYS20021_4', 'BYS20021_5', 'BYS20021_6', 'BYS20021_7', 'BYS20021_8', 'BYS20021_9', 'BYS20021_10', 'BYS20021_11', 'BYS20021_12', 'BYS20021_13', 'BYS22002_1', 'BYS22002_2', 'BYS22002_3', 'BYS22002_4', 'BYS22002_5', 'BYS22002_6', 'BYS22002_7', 'BYS22002_8', 'BYS22002_9', 'BYS22002_10', 'BYS23003_1', 'BYS23003_2', 'BYS23003_3', 'BYS23003_4', 'BYS23003_5', 'BYS23003_6', 'BYS23003_7', 'BYS23003_8', 'BYS23003_9', 'BYS04009_1', 'BYS04009_2', 'BYS04009_3', 'BYS04009_4', 'BYS04009_5', 'BYS04009_6', 'BYS04009_7', 'BYS04009_8', 'BYS04009_9', 'BYS09003_1', 'BYS09003_2', 'BYS09003_3', 'BYS09003_4', 'BYS09003_5', 'BYS09003_6', 'BYS08007_1', 'BYS08007_2', 'BYS08007_3', 'BYS08007_4', 'BYS08007_5', 'BYS08007_6', 'BYS08007_7', 'BYS08007_8', 'BYS08007_9', 'BYS08007_10', 'BYS08007_11', 'BYS08007_12', 'BYS08007_13', 'BYS08007_14']
data=result_merge[must_include_cols]

target = df['J007C']

print('-----------------------------------------------------------------------')
X_train, X_test, y_train, y_test=train_test_split(data,target, random_state=7)
lr_clf=LogisticRegression(solver='lbfgs', multi_class='auto')
lr_clf.fit(X_train, y_train)
print(lr_clf.score(X_train, y_train))
print(lr_clf.score(X_test, y_test))
total_predict=[]
for i in range(100):
    total_predict.append(lr_clf.predict([X_train.values[i]]))

predict = lr_clf.predict([X_test.values[0]])
# print(recall_score(predict, y_test.values[0]))
# print(precision_score(predict,y_test.values[0]))


# # poly
#
# X_train_poly=poly.fit_transform(X_train)
# X_test_poly=poly.transform(X_test)
# lr_clf.fit(X_train_poly, y_train)
# print(lr_clf.score(X_train_poly, y_train))
# print(lr_clf.score(X_test_poly, y_test))


# RandomForest
print('-------------------------RandomForestClassifier----------------------------------')
forest=RandomForestClassifier(n_estimators=200, random_state=7, max_depth=10)
forest.fit(X_train, y_train)
print(forest.score(X_train, y_train))
print(forest.score(X_test,y_test))
print('-------------------------------------------------------------')
# forest2=RandomForestClassifier(n_estimators=5, random_state=7)
# forest2.fit(X_train_poly, y_train)
# print(forest2.score(X_train_poly, y_train))
# print(forest2.score(X_test_poly,y_test))


#ridge

# from sklearn.linear_model import Ridge
# print('---------------------------------------------------------')
# ridge2=Ridge().fit(X_train_poly, y_train)
# print(ridge2.score(X_train_poly, y_train))
# print(ridge2.score(X_test_poly, y_test))