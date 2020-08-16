from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix



df=pd.read_csv('data.csv')

target=df.iloc[:,-1]
data=df.iloc[:,1:-1]
X_trainVal, X_test, y_trainVal, y_test=train_test_split(data, target, random_state=7)
X_train,X_valid,y_train, y_valid=train_test_split(X_trainVal,y_trainVal)
best_score=0
for iteration in [5,8,10,100]:
    for learningRating in [0.01,0.5,0.1,1]:
        for depth in [3,4,5,6,10]:
            cb=CatBoostClassifier(iterations=iteration, learning_rate=learningRating,depth=depth,loss_function='MultiClass')
            scores=cross_val_score(cb, X_trainVal, y_trainVal, cv=5)
            score=np.mean(scores)
            if score > best_score:
                best_score=score
                best_parameter={'iteration: ' : iteration, 'learing_rate: ':learningRating, 'depth: ': depth}

print(best_parameter)
