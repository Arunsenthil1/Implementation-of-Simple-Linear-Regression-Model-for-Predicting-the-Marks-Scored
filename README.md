# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.
```
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Sushmitha Gembunathan
RegisterNumber:  212224040342
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print("df.head()")
print()
print(df.head())
print()
print()
print("df.tail()")
print()
print(df.tail())
x = df.iloc[:,:-1].values
print("Array of X")
print()
print(x)
print()
y = df.iloc[:,1].values
print("Array of Y")
print()
print(y)
print()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
<img width="778" height="384" alt="Screenshot 2025-08-29 140632" src="https://github.com/user-attachments/assets/6f7bfc8d-b92e-4a5c-bc29-1a7ea6152715" />
<img width="974" height="701" alt="Screenshot 2025-08-29 140641" src="https://github.com/user-attachments/assets/3a595158-49de-48e0-9a2d-32fc3724234b" />
<img width="804" height="585" alt="Screenshot 2025-08-29 140653" src="https://github.com/user-attachments/assets/628a43c3-6727-4ef4-8644-cdebf19018ec" />
<img width="788" height="623" alt="Screenshot 2025-08-29 140702" src="https://github.com/user-attachments/assets/e807bb5d-2681-417b-a2f1-3046105f023d" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
