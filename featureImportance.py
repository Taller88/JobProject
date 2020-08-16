import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
model=ExtraTreesClassifier()
df=pd.read_csv('Real_final.csv')
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
target=df['J007C']
data=df.iloc[:, 5:-1]
X_train, X_test, y_train, y_test=train_test_split(data, target, random_state=7)

# model.fit(data, target)
rf=RandomForestClassifier()
rf.fit(X_train, y_train)
importances=rf.feature_importances_

(pd.Series(rf.feature_importances_, index=data.columns)
   .nlargest(4)
   .plot(kind='barh'))
# plt.barh(range(len(importances)),importances)
# plt.show()
# feat_importance=pd.Series(model.feature_importances_, index=total_data.columns)
# plt.figure(figsize=(10,50))
# feat_importance.plot(kind='barh')
# plt.show()

