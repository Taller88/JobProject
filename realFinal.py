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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import xgboost as xg
mpl.rcParams['axes.unicode_minus']=False
mpl.rcParams.update({'font.size':14,'font.weight':'bold'})

font_name=fm.FontProperties(fname="c:\\windows\\fonts\\malgun.ttf").get_name()
rc('font',family=font_name)

a="21113131533333236"
predict_data=[]
for i in a:
    predict_data.append(i)


df=pd.read_csv('data.csv')

target=df.iloc[:,-1]
data=df.iloc[:,1:-1]
X_trainval, X_test, y_trainval, y_test=train_test_split(data, target, random_state=7)
cb=CatBoostClassifier(iterations=8, learning_rate=0.1,depth=6,loss_function='MultiClass')
cb.fit(X_trainval,y_trainval)
rf=RandomForestClassifier()
rf.fit(X_trainval, y_trainval)
xgb=xg.XGBClassifier(objective ='multi:softmax', max_depth=5)

print(cb.score(X_trainval,y_trainval))
print(cb.score(X_test, y_test))

# print(cb.predict_proba(predict_data))
y_pred=cb.predict(X_test)
print(classification_report(y_test,y_pred,target_names=['1','2','3','4']))

xgb.fit(X_trainval, y_trainval)
xgb_predict=pd.DataFrame(xgb.predict(X_test))

xgb_predict.to_csv('xgb.csv', index=False)
print("XGBoosting Confusion Matrix")
print(classification_report(y_test, xgb_predict,target_names=['1','2','3','4']))


rf_predict=pd.DataFrame(rf.predict(X_test))

print('RandomForest Confusion Matrix')
rf_predict.to_csv('RandomForest.csv', index=False)
print(classification_report(y_test, rf_predict,target_names=['1','2','3','4']))
y_test.to_csv('y_test.csv', index=False)