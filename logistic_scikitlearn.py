import pandas as pd
import numpy as np
Diabetes=pd.read_csv('diabetes.csv')
table1=np.mean(Diabetes,axis=0)
table2=np.std(Diabetes,axis=0)

inputData=Diabetes.iloc[:,:8]
outputData=Diabetes.iloc[:,8]

from sklearn.linear_model import LogisticRegression
logit1=LogisticRegression()
logit1.fit(inputData,outputData)

logit1.score(inputData,outputData)

##True positive
trueInput=Diabetes.ix[Diabetes['Outcome']==1].iloc[:,:8]
trueOutput=Diabetes.ix[Diabetes['Outcome']==1].iloc[:,8]
##True positive rate
pred = logit1.predict(trueInput)
print('True positives prediction: ', pred)
np.mean(pred==trueOutput)
##Return around 55%

##True negative
falseInput=Diabetes.ix[Diabetes['Outcome']==0].iloc[:,:8]
falseOutput=Diabetes.ix[Diabetes['Outcome']==0].iloc[:,8]
##True negative rate
pred = logit1.predict(falseInput)
print('False positives prediction: ', pred)
np.mean(pred==falseOutput)
##Return around 90%

###Confusion matrix with sklearn
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
confusion_matrix(logit1.predict(inputData),outputData)
