import pandas as pd
import numpy as np
import xgboost as xg
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel


df = pd.read_csv('resultVariable.csv')

categorical_names=['h_major','col_major','sch_code','BYS04008', 'BYS04009', 'BYS07002', 'BYS07006', 'BYS07014', 'BYS08001', 'BYS08002', 'BYS08005', 'BYS09002', 'BYS09003', 'BYS22001','BYS22002','BYS23002', 'BYS23003', 'BYS23004', 'BYS24002', 'BYS16001', 'BYS20021', 'BYS27020','BYS08007']
numerical_names=['F8Y140_Influence', 'F8Y140_Feeling', 'F8Y140_consideration', 'F8Y140_Intuition', 'F8Y140_Sensing','BYS02003', 'BYS02004', 'BYS02012', 'BYS04007', 'BYS07001', 'BYS07025', 'BYS07026', 'BYS07027', 'BYS07029', 'BYS08003', 'BYS08004', 'BYS09001', 'BYS09004', 'BYS10001', 'BYS10002', 'BYS10003', 'BYS10005', 'BYS11001', 'BYS11002', 'BYS11003', 'BYS11004', 'BYS11005', 'BYS11006', 'BYS13001', 'BYS22003', 'BYS22006', 'BYS22012', 'BYS22013', 'BYS22014', 'BYS22015', 'BYS22017', 'BYS25001', 'BYS25003']
discrete_names=['GENDER', 'BYS03001', 'BYS04010', 'BYS05001', 'BYS05003', 'BYS05006', 'BYS06001', 'BYS06004', 'BYS07003', 'BYS07022', 'BYS12001', 'BYS12004', 'BYS12007', 'BYS12010', 'BYS12013', 'BYS12016', 'BYS12019', 'BYS12022', 'BYS14001', 'BYS14013', 'BYS14025', 'BYS15001', 'BYS20001', 'BYS20007', 'BYS20009', 'BYS20011', 'BYS20013', 'BYS20015', 'BYS20017', 'BYS20018', 'BYS20019', 'BYS20020', 'BYS21001', 'BYS22010', 'BYS24001', 'BYS24005', 'BYS26001', 'BYS26002', 'BYS26003', 'BYS26004', 'BYS26005', 'BYS26006', 'BYS26007', 'BYS26009', 'BYS27001', 'BYS27003', 'BYS27006', 'BYS27012']



categorical_data = df[categorical_names].astype('str')
onehot_data = pd.get_dummies(categorical_data)

onehot_data['BYSID']=df['BYSID']
numerical_data = df[numerical_names]
numerical_data['BYSID']=df['BYSID']
discrete_data = df[discrete_names]
discrete_data['BYSID']=df['BYSID']

total_data=pd.merge(onehot_data, numerical_data, on='BYSID')
total_data=pd.merge(total_data,discrete_data, on='BYSID')

del total_data['BYSID']
# total: 전체 데이터

print('--------------------------------------------------------------------------------------------------')
mustHave_columns=['F8Y140_Influence', 'F8Y140_Feeling', 'F8Y140_consideration', 'F8Y140_Intuition', 'F8Y140_Sensing']
need_oneHot=[]
numeric=[]
for i in mustHave_columns:
    if i in categorical_names:
        need_oneHot.append(i)
    else:
        numeric.append(i)
if need_oneHot:
    selected_oneHot=pd.get_dummies(categorical_data[need_oneHot])
    selected_oneHot['BYSID']=df['BYSID']
    selected_numeric = numerical_data(numeric)
    selected_numeric['BYSID']=df['BYSID']
    selected_data=pd.merge(selected_numeric, selected_oneHot, on='BYSID')
else:
    selected_numeric=numerical_data[numeric]
    selected_data=selected_numeric
# selected_data: 추출한 독립변수
data=selected_data
print(data)
target=df['J007C']
print(target)
X_train, X_test, y_train, y_test=train_test_split(total_data, target, test_size=0.2)
#  - multi:softmax : softmax를 사용한 다중 클래스 분류, 예측된 클래스를 반환한다. (not probabilities)
xgb=xg.XGBClassifier(objective ='multi:softmax', max_depth=5)
xgb.fit(X_train, y_train)
xgb.feature_importances_
print(xgb.score(X_train, y_train))
print(xgb.score(X_test, y_test))
print(xgb.predict(X_test))

#  - multi:softprob : softmax와 같지만 각 클래스에 대한 예상 확률을 반환한다.
xgb2=xg.XGBClassifier(objective ='multi:softprob', max_depth=10)
xgb2.fit(X_train, y_train)
print(xgb2.score(X_train, y_train))
print(xgb2.score(X_test, y_test))

print(xgb2.predict_proba(X_test))
sel = SelectFromModel(xg.XGBClassifier(objective ='multi:softprob', max_depth=10))
sel.fit(X_train, y_train)

sel.get_support()
selected_feat= X_train.columns[(sel.get_support())]
print(len(selected_feat))

best_feature=selected_feat.tolist()
print(best_feature)
predict_xgb = xgb2.predict(X_test)
predict_xgb.to_csv('xgb.csv', index=False)
y_test.to_csv('y_test.csv', index=False)