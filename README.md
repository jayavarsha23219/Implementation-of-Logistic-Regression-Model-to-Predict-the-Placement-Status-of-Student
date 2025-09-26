# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices. 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: JAYAVARSHA T
RegisterNumber: 212223040075 
*/
```
```
import pandas as pd
data=pd.read_csv("/content/Placement_Data.csv")
data.head()
```
<img width="993" height="162" alt="image" src="https://github.com/user-attachments/assets/aa394f1d-15dc-4523-b272-b47d30b1d297" />

```
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Removes the specified row or column
data1.head()
```
<img width="850" height="156" alt="image" src="https://github.com/user-attachments/assets/b44e12a4-8c6d-48d2-88ce-423343568401" />

```
data1.isnull().sum()
```
<img width="258" height="407" alt="image" src="https://github.com/user-attachments/assets/c5bfd6e5-e773-4d9f-9c77-dee94f9123bd" />

```
data1.duplicated().sum()
```
<img width="118" height="27" alt="image" src="https://github.com/user-attachments/assets/0ad6d604-5d78-4254-bd49-e0a9e25413c5" />

```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
```
<img width="866" height="350" alt="image" src="https://github.com/user-attachments/assets/0f4a915a-231c-4787-a0f7-15825eef8805" />

```
x=data1.iloc[:,:-1]
x
```
<img width="808" height="341" alt="image" src="https://github.com/user-attachments/assets/c5644193-3d36-4b72-8977-15e037958c8e" />

```
y=data1["status"]
y
```
<img width="166" height="312" alt="image" src="https://github.com/user-attachments/assets/d7a106ac-fe09-4e34-b0e4-ccf05d823dcf" />

```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")# A library for large linear classification
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred
```
<img width="512" height="33" alt="image" src="https://github.com/user-attachments/assets/64621260-84f9-4e12-992d-97819468e2fd" />

```
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)# Accuracy Score = (TP+TN)/
#accuracy_score(y_true,y_prednormalize=False)
accuracy
```
<img width="156" height="23" alt="image" src="https://github.com/user-attachments/assets/7222a72e-2fd8-4a5b-a0c4-046d182a452d" />

```
from sklearn.metrics import confusion_matrix
confusion = (y_test,y_pred)
confusion
```
<img width="515" height="671" alt="image" src="https://github.com/user-attachments/assets/11771dce-08b1-40d6-a6c3-03baee56ee2c" />

```
from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)
```
<img width="432" height="137" alt="image" src="https://github.com/user-attachments/assets/211106d8-2d46-45fc-9eba-64382025cb1b" />


## Output:
![the Logistic Regression Model to Predict the Placement Status of Student](sam.png)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
