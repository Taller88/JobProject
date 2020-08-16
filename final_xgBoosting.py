import numpy as np
import pandas as pd
import xgboost as xg
from pandas.core.common import SettingWithCopyWarning
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings(action='ignore', category=SettingWithCopyWarning)

df=pd.read_csv('data.csv')


categorical_names = ['father_job','BYS22001','BYS22002','BYS23002','BYS04008','BYS04009','selfConcept_cate', 'lifeSelfConfidence_cate', 'emotionalStability_cate', 'schFacilSatisfi_cate', 'careerSelfEfficacy_cate', 'liberalArts_cate', 'naturalSciences_cate']
discrete_names=['AT_05','GENDER','teacherRel_cate']
discrete_data=df[discrete_names]
discrete_data['BYSID']=df['BYSID']
categorical_data=df[categorical_names]

categorical_data=pd.get_dummies(categorical_data.astype('str'))
categorical_data['BYSID']=df['BYSID']
data=pd.merge(discrete_data,categorical_data, on='BYSID')
del data['BYSID']

target=df['J007C'].astype('str')

# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ

X_train, X_test, y_train, y_test= train_test_split(data, target, random_state=7)
X_train=X_train.values
X_test=X_test.values

xgb=xg.XGBClassifier(objective ='multi:softmax', max_depth=4,learning_rate=0.1, n_estimators=10)
xgb.fit(X_train, y_train)
print(xgb.score(X_train, y_train))
print(xgb.score(X_test, y_test))
