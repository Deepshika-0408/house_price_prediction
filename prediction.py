#importing the dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection  import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
#importing  the dataset
house_price_dataframe=pd.read_csv("boston.csv")
#print first 5 rows
print(house_price_dataframe.head())
#since the dataset is downloaded from kaggle no need to add new column called price as it is added,but if it is downloaded from sklearn we should do extra work
#lets check number of rows and columns in dataframe
print(house_price_dataframe.shape)
#now we need to check if the dataset has any missing values
#we need to procees steps for misssing values
print(house_price_dataframe.isnull().sum())
#statistical measures of the dataframe
print(house_price_dataframe.describe())
#now we need to do further data analysis to find the corellation between the data
#positive corelation/negative corelation
correlation=house_price_dataframe.corr()
#constructing a heatmap to  show the correlation between the data
plt.figure(figsize=(10,10))
sns.heatmap(correlation, annot=True, cmap='Blues', square=True,cbar=True,fmt='.1f',annot_kws={'size':8})
plt.show()
#now we need to splitting data into data and label
#in this we use x with all data and y with all prices
X=house_price_dataframe.drop(['PRICE'],axis=1)
Y=house_price_dataframe['PRICE']
print(X)
print(Y)
#splitting the data into train and test data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=2)
print(X.shape,X_test.shape,X_train.shape)
#model training --xgboost regressor
model=XGBRegressor()
model.fit(X_train,Y_train)
#evaluate the model--prediction
Y_pred=model.predict(X_train)
#r squared error
# score_1=metrics.r2_score(Y_train,Y_pred)
# score_2=metrics.mean_absolute_error(Y_train,Y_pred)
# print(score_1)
# print(score_2)
test_data_pred=model.predict(X_test)
score_1=metrics.r2_score(Y_test,test_data_pred)
score_2=metrics.mean_absolute_error(Y_test,test_data_pred)
print(score_1)
print(score_2)
#visualize the predicted values
plt.scatter(Y_train,Y_pred)
plt.xlabel("actual prices")
plt.ylabel("predicted prices")
plt.title("actal vs predicted prices")
plt.show()

