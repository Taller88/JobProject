from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
import graphviz
from sklearn.tree import export_graphviz
import pydotplus
from IPython.display import Image

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
names=df.columns.tolist()
dt= tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
X_train, X_test, y_train, y_test=train_test_split(data, target, random_state=7)

dt.fit(X_train, y_train)
print(dt.score(X_train, y_train))
print(dt.score(X_test, y_test))
dot_data = export_graphviz(dt, out_file=None,feature_names=data.columns.tolist(),class_names=['Management','Leisure','Sales','Production'], filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())

#
# export_graphviz(dt,out_file='dicsionTree1.dot',class_names=['Management','Leisure','Sales', 'Production'],impurity=False,filled=True,feature_names=names)
# (graph,)=pydot.graph_from_dot_file('dicisionTree1.dot', encoding='utf8')
# graph.write_png('dicisionTree1.png')
# DecsionTreeTest1

# export_graphviz(dt, out_file='tree.dot', class_names=['Management','Leisure','Sales', 'Production'],impurity=False, filled=True)
# with open('tree.dot', encoding='utf8') as f:
#     dot_graph = f.read()
# print(graphviz.Source(dot_graph))
