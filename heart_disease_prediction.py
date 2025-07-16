# -*- coding: utf-8 -*-
"""
Created on Tue May 13 15:51:55 2025

@author: supri
"""

import numpy as np #numpy used for make numpy arrays that is useful for preprocessing
import pandas as pd #pandas is used for creating data frames
from sklearn.preprocessing import StandardScaler #standardize the data to a common range 
from sklearn.model_selection import train_test_split #split our data into training or testing data
from sklearn import svm 
from sklearn.metrics import accuracy_score
#Data collection and analysis\
diabetes_dataset = pd.read_csv("C:\\Users\\supri\\Downloads\\diabetes.csv")
#pd.read_csv?
diabetes_dataset.head()#printing first five rows of dataset
diabetes_dataset.shape#No of rows and columns in our dataset
diabetes_dataset.describe()#getting the statistical measures of the data
diabetes_dataset['Outcome'].value_counts()#gives the count of outcomes 
#0--> non-diabetic
#1--> diabetic
diabetes_dataset.groupby('Outcome').mean()#gives the mean value for each parameters for different outcomes
X= diabetes_dataset.drop(columns="Outcome",axis=1) #seperating the data and labels where axis 1 mean columns 
Y= diabetes_dataset["Outcome"]
print(X)
print(Y)
print(diabetes_dataset)
scaler=StandardScaler() #data standardization
scaler.fit(X)
standardized_data= scaler.transform(X)
print(standardized_data)
X= standardized_data
Y= diabetes_dataset['Outcome']
print(X)
print(Y)
# Train Test Split
X_train, X_test, Y_train, Y_test =train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=2)
#test_size=0.2 represents 20% of test data on basis Y so that similar proportion of data will be considered for training and testing
#random state 
print(X_train)
print(X_train.shape)
#Training the model
classifier=svm.SVC(kernel='linear')
#Training the svm classifier
classifier.fit(X_train, Y_train)
#Model Evaluation and accuracy score
X_train_prediction = classifier.predict(X_train)
training_data_accuracy =accuracy_score(X_train_prediction,Y_train)
print("Accuracy score of training data :",training_data_accuracy)
X_test_prediction = classifier.predict(X_test)
testing_data_accuracy =accuracy_score(X_test_prediction,Y_test)
print("Accuracy score of testing data :",testing_data_accuracy)


#Making a prediction system
Pregnancies =int(input("enter no. of pregnency:"))	
Glucose =int(input("enter your glucose level:"))	
BloodPressure =int(input("enter your blood pressure level:"))	
SkinThickness=int(input("enter your skin thickness:"))	
Insulin	=int(input("enter your insulin level:"))
BMI	=float(input("enter your body mass index:"))
DiabetesPedigreeFunction=float(input("enter diabetes pedigree function:"))	
Age=int(input("enter your age:"))

input_data =(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)
print(input_data)
#cahnging the input data as numpy array
input_data_as_numpy_array =np.asarray(input_data)

#reshape the array as we are predicting for one instance
input_data_reshaped =input_data_as_numpy_array.reshape(1, -1)

#now standardize these data as we did for training data
std_data =scaler.transform(input_data_reshaped)
print(std_data)

prediction=classifier.predict(std_data)

print(prediction)

if(prediction==0):
    print("The person is not diabetic")
else:
    print("The person is diabetic")
    
