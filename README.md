# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the required packages.
2. Read the given dataset and assign x and y array.
3. Split x and y into training and test set.
4. Scale the x variables.
5. Fit the logistic regression for the training set to predict y.
6. Create the confusion matrix and find the accuracy score, recall sensitivity and specificity
7. Plot the training set results.


## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: R.Rajalakshmi
RegisterNumber:  212219040116
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#reading and displaying dataframe
df=pd.read_csv("Social_Network_Ads (1).csv")
df
x=df.iloc[:,[2,3]].values
y=df.iloc[:,4].values 
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=0)
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
xtrain=sc_x.fit_transform(xtrain)
xtest=sc_x.fit_transform(xtest)
from sklearn.linear_model import LogisticRegression
c=LogisticRegression(random_state=0)
c.fit(xtrain,ytrain)
ypred=c.predict(xtest)
ypred
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(ytest,ypred)
cm
from sklearn import metrics
acc=metrics.accuracy_score(ytest,ypred)
acc
r_sens=metrics.recall_score(ytest,ypred,pos_label=1)
r_spec=metrics.recall_score(ytest,ypred,pos_label=0)
r_sens,r_spec
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
xs,ys=xtrain,ytrain
x1,x2=np.meshgrid(np.arange(start=xs[:,0].min()-1,stop=xs[:,0].max()+1,step=0.01),
               np.arange(start=xs[:,1].min()-1,stop=xs[:,1].max()+1,step=0.01))
plt.contourf(x1,x2,c.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
                            alpha=0.75,cmap=ListedColormap(('skyblue','green')))
plt.xlim(x1.min(),x2.max())
plt.ylim(x2.min(),x1.max())
for i,j in enumerate(np.unique(ys)):
    plt.scatter(xs[ys==j,0],xs[ys==j,1],
                c=ListedColormap(('black','white'))(i),label=j)
plt.title("Logistic Regression(Training Set)")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()
 
```

## Output:

## Dataset:
![Screenshot (126)](https://user-images.githubusercontent.com/87656716/173222287-92676286-7153-4045-9921-d70c7db50b36.png)

## Predicted y array:
![Screenshot (128)](https://user-images.githubusercontent.com/87656716/173222332-24884ba2-8b20-4eaa-a3c0-66ff0eeb2a41.png)

## Confusion matrix:
![Screenshot (130)](https://user-images.githubusercontent.com/87656716/173222392-d9773134-7608-4185-8c4f-46b672f87e41.png)

## Accuracy score:
![Screenshot (133)](https://user-images.githubusercontent.com/87656716/173222661-d377a203-9395-4faf-89fe-d354ca9f6efa.png)

## Recall sensitivity and specificity:
![Screenshot (134)](https://user-images.githubusercontent.com/87656716/173222728-20ea752a-1d01-4c3f-bc53-1b215714b19b.png)

## Logistic Regression graph:
![Screenshot (136)](https://user-images.githubusercontent.com/87656716/173222867-2cd8904e-63d3-4186-86cd-908a5f8e56b3.png)




## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

