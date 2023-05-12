# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Prepare your data

    Clean and format your data
    Split your data into training and testing sets

2.Define your model

    Use a sigmoid function to map inputs to outputs
    Initialize weights and bias terms

3.Define your cost function

    Use binary cross-entropy loss function
    Penalize the model for incorrect predictions

4.Define your learning rate

    Determines how quickly weights are updated during gradient descent

5.Train your model

    Adjust weights and bias terms using gradient descent
    Iterate until convergence or for a fixed number of iterations

6.Evaluate your model

    Test performance on testing data
    Use metrics such as accuracy, precision, recall, and F1 score

7.Tune hyperparameters

    Experiment with different learning rates and regularization techniques

8.Deploy your model

    Use trained model to make predictions on new data in a real-world application.

## Program:
```py
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: ALdrin Lijo J E
RegisterNumber: 212222240007
*/

import pandas as pd
data=pd.read_csv("Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
## Output:
### Initial data set:

![image](https://github.com/aldrinlijo04/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118544279/a6ec12a8-e47b-45f6-b8e2-ec4c81b8db82)
### Data info:

![image](https://github.com/aldrinlijo04/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118544279/8aa44bf1-6760-4b58-aa9d-7384155a7952)
### Optimization of null values:

![image](https://github.com/aldrinlijo04/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118544279/8e542229-3120-437d-9812-99bec91e6acf)
### Assignment of x and y values:

![image](https://github.com/aldrinlijo04/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118544279/b9e4fc04-f357-406d-837f-03fe5f8ba8fd)
![image](https://github.com/aldrinlijo04/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118544279/d4277cac-066c-43f2-b2b7-1bd4dcaa5bde)
### Converting string literals to numerical values using label encoder:

![image](https://github.com/aldrinlijo04/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118544279/0f19677a-a191-4757-891e-ca46fa596705)
### Accuracy:
![image](https://github.com/aldrinlijo04/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118544279/b44d5167-3f12-40ce-91a8-cdd49fe3987c)
### Prediction:
![image](https://github.com/aldrinlijo04/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118544279/9d4d1b13-fb3e-4554-bee5-700764312ea8)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
