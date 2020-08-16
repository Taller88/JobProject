import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.linear_model import RidgeClassifier
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rc
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import cross_val_score

mpl.rcParams['axes.unicode_minus']=False
mpl.rcParams.update({'font.size':14,'font.weight':'bold'})

font_name=fm.FontProperties(fname="c:\\windows\\fonts\\malgun.ttf").get_name()
rc('font',family=font_name)

a="21113131533333236"
predict_data=[]
for i in a:
    predict_data.append(i)

dict_ex={0:(0.1,0,0,0),1:(0,0.1,0,0),2:(0,0,0.1,0),3:(0,0,0,0.1)}

df=pd.read_csv('data.csv')
#
# categorical_names = ['father_job','col_major','BYS22001','BYS22002','BYS23002','BYS04008','BYS04009','selfConcept_cate', 'lifeSelfConfidence_cate', 'emotionalStability_cate', 'schFacilSatisfi_cate', 'careerSelfEfficacy_cate', 'liberalArts_cate', 'naturalSciences_cate']
# discrete_names=['AT_05','GENDER','teacherRel_cate']
# discrete_data=df[discrete_names]
# discrete_data['BYSID']=df['BYSID']
# categorical_data=df[categorical_names]
#
# categorical_data=pd.get_dummies(categorical_data.astype('str'))
# categorical_data['BYSID']=df['BYSID']
# data=pd.merge(discrete_data,categorical_data, on='BYSID')
# print(data)

target=df.iloc[:,-1]
data=df.iloc[:,1:-1]
X_trainval, X_test, y_trainval, y_test=train_test_split(data, target, random_state=7)
cb=CatBoostClassifier(iterations=8, learning_rate=0.1,depth=6,loss_function='MultiClass')
cb.fit(X_trainval,y_trainval)
print(cb.score(X_trainval,y_trainval))
print(cb.score(X_test, y_test))
fi=cb.feature_importances_
feat_importance=pd.Series(fi, index=data.columns,)
print(data)

# result=cb.predict(X_test.values[1])
# label='경영, 사무, 금융, 공공','미용, 여행, 음식','영업, 판매, 운송직','기술, 정비, 생산직'
# sizes=cb.predict_proba(predict_data)
# print(result[0][0])
# explode=dict_ex[result[0][0]-1]
# plt.figure(figsize=(14,7))
# plt.pie(sizes,explode=explode,labels=label,counterclock=False, autopct='%1.1f%%', shadow=True,startangle=90)
# plt.axis('equal')
#
# plt.legend(label, loc="right", bbox_transform=plt.gcf().transFigure)
#
# plt.savefig('pieChart.jpg')
#
