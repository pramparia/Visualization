# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 18:51:48 2023

@author: pooja
"""

import pandas as pd
import seaborn as sn
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('C:\Pooja\Masters/OPER-5151EL-01-Dataset_Final_Exam_SP23.csv')

y=data.loc[:,['response_binary']]
x=data.loc[:,data.columns.isin(['age','balance','day','duration','campaign','pdays','previous'])]

reg = LinearRegression()
reg.fit(x,y)
predic_class=reg.predict(x)

data['Pred_reg']=predic_class
dis_score_fraud = data.loc[data['response_binary']==1,'Pred_reg'].mean()
dis_score_good = data.loc[data['response_binary']==0,'Pred_reg'].mean()
dis_score = 0.5*(dis_score_good + dis_score_fraud)

Pred_class = []

for i in range(len(y)):
    if data['Pred_reg'][i]>=dis_score :
        Pred_class.append(1)
    else:
        Pred_class.append(0)
        
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y,Pred_class)

n,m = conf_matrix.shape
conf_mat_rel = np.zeros((n, m))

for i in range(n):
    for j in range(m):
        conf_mat_rel[i][j]=np.round(conf_matrix[i][j]/sum(conf_matrix[i]),5)

conf_cm = pd.DataFrame(conf_mat_rel, range(2))
fig = plt.figure()
sn.heatmap(conf_cm, annot=True,fmt=".2%")
plt.show()
fig.savefig('Final exam confusion heatmap.pdf')