import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

total=pd.read_csv('realTotal2.csv')

df=total.values
data=df[:,:-1]
target=df[:,-1]
print(data.shape)
print(target.shape)
