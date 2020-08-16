import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

df=pd.read_csv('total_onehot_fourth.csv')
data = df.iloc[:, :-1]
target=df['J007C']
print(data.shape)
X_train, X_test, y_train, y_test=train_test_split(data, target, random_state=1)
poly=PolynomialFeatures(degree=2, interaction_only=True)
X_train_poly=poly.fit_transform(X_train)
X_test_poly=poly.transform((X_test))
rf=RandomForestClassifier(n_estimators=4)
rf.fit(X_train, y_train)
rf2=RandomForestClassifier(n_estimators=4)
rf2.fit(X_train_poly, y_train)

ridge=Ridge().fit(X_train, y_train)


print(rf.score(X_train, y_train))
print(rf.score(X_test, y_test))
print('--------------------------------------')
print(rf2.score(X_train_poly, y_train))
print(rf2.score(X_test_poly, y_test))
print('--------------------------------------')
print(ridge.score(X_train, y_train))
print(ridge.score(X_test, y_test))

print('--------------------------------------')
