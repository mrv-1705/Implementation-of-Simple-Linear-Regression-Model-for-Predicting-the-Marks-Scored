# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1. Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.
```

## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Kurapati Vishnu Vardhan Reddy
RegisterNumber:  212223040103
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
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
1.![image](https://github.com/user-attachments/assets/4cf6c1e8-9483-47a5-bd70-495f8964a93e)

2.![image](https://github.com/user-attachments/assets/16ed5647-4eb9-413a-a17d-bee25ac3cc59)

3.![image](https://github.com/user-attachments/assets/ed5a8af4-ba33-4bc9-b6ac-35b05c713375)


4.![image](https://github.com/user-attachments/assets/fc6103a8-f638-4bb4-a30e-0e5438605900)


5.![image](https://github.com/user-attachments/assets/41709d17-d0c3-471e-b32f-6f6aeb400d0e)


6.![image](https://github.com/user-attachments/assets/8a70c6c2-cdf7-46fc-8ca4-78ea8ecc219b)


7.![image](https://github.com/user-attachments/assets/d82a8283-4e8a-4dab-8717-6695dde07624)




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
