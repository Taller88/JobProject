import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import numpy as np
import xgboost as xg
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.preprocessing import MinMaxScaler
lr_clf=LogisticRegression(solver='lbfgs', multi_class='multinomial')
rf=RandomForestClassifier(n_estimators=10)
poly=PolynomialFeatures(degree=4, interaction_only=True)
df=pd.read_csv('real_bysid.CSV')
data=df.iloc[:,1:-1].astype('str')
print(df.columns.tolist())


#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ

categorical_names=['father_job','col_major','BYS22001','BYS22002','BYS23002','BYS04008','BYS04009','selfConcept_cate', 'lifeSelfConfidence_cate', 'emotionalStability_cate', 'schFacilSatisfi_cate', 'careerSelfEfficacy_cate', 'liberalArts_cate', 'naturalSciences_cate']
discrete_names=['AT_05','GENDER','teacherRel_cate']

discrete_data=df[discrete_names]
discrete_data['BYSID']=df['BYSID']
categorical_data=pd.get_dummies(df[categorical_names])

categorical_data['BYSID']=df['BYSID']
data=pd.merge(discrete_data,categorical_data, on='BYSID')
del data['BYSID']
print(data)

#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
target=df['J007C'].astype('str')
print(data.columns.tolist())
print(target.shape)
X_train, X_test, y_train,y_test=train_test_split(data,target, random_state=7, test_size=0.3)
lr_clf.fit(X_train, y_train)
print("Logistic Train Score: ", lr_clf.score(X_train, y_train))
print("Logistic Test Score: ", lr_clf.score(X_test, y_test))
print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
y_pred=lr_clf.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test,y_pred))

X_train_poly=poly.fit_transform(X_train)
X_test_poly=poly.transform(X_test)
lr_clf.fit(X_train_poly, y_train)
print("Logistic+Polynomial Train: ", lr_clf.score(X_train_poly , y_train))
print("Logistic+Polynomial Train: ", lr_clf.score(X_test_poly , y_test))


ridge=Ridge(normalize=True)
ridge.fit(X_train_poly,y_train)
print("Ridge 일반 Train: ", ridge.score(X_train_poly,y_train))
print("Ridge 일반 Test: ", ridge.score(X_test_poly,y_test))

# ridge1=Ridge()
# ridge1.fit(X_train_poly,y_train)
# print("Ridge poly Train: ", ridge.score(X_train_poly,y_train))
# print("Ridge poly Test: ", ridge.score(X_test_poly,y_test))


rf.fit(X_train,y_train)
print("RandomForest Score: ", rf.score(X_train, y_train))
print("RandomForest Score: ", rf.score(X_test, y_test))
print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡSVMㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')
# svm=SVC()
# svm.fit(X_train, y_train)
# print('SVM Score: ', svm.score(X_train, y_train))
# print('SVM Score: ', svm.score(X_test, y_test))


print('ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡXGBoostinㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ')


stScaler=StandardScaler()
stScaler.fit(X_train)
st_X_train=stScaler.transform(X_train)
st_X_test=stScaler.transform(X_test)

sel = SelectFromModel(RandomForestClassifier(n_estimators=4, max_features=10))
sel.fit(X_train, y_train)

sel.get_support()
selected_feat= X_train.columns[(sel.get_support())]
print(len(selected_feat))
print(selected_feat)


xgb=xg.XGBClassifier(objective ='multi:softmax', max_depth=5)
xgb.fit(X_train, y_train)
print(xgb.score(X_train, y_train))
print(xgb.score(X_test, y_test))
print(xgb.predict(X_test))

print("*****************************************************")
y_pred=xgb.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test,y_pred))
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

rs_scaler=RobustScaler()
rs_X_train=rs_scaler.fit_transform(X_train)
rs_X_test=rs_scaler.transform(X_test)

minmax=MinMaxScaler()
mm_X_train=minmax.fit_transform(X_train)
mm_X_test=minmax.transform(X_test)


# ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ

knn=KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train, y_train)
print(knn.score(X_train,y_train))
print(knn.score(X_test,y_test))

y_pred=knn.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test,y_pred))

knn=KNeighborsClassifier(n_neighbors=4)
knn.fit(rs_X_train, y_train)
print(knn.score(rs_X_train,y_train))
print(knn.score(rs_X_test,y_test))

y_pred=knn.predict(rs_X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test,y_pred))

knn=KNeighborsClassifier(n_neighbors=4)
knn.fit(mm_X_train, y_train)
print(knn.score(mm_X_train,y_train))
print(knn.score(mm_X_test,y_test))

y_pred=knn.predict(mm_X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test,y_pred))
