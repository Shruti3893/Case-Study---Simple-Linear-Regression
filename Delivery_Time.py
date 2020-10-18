# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 17:50:16 2020

@author: Lenovo
"""

#Problem Statement : Do the necessary transformations for input variables for getting 
#better R^2 value for the model prepared.
#2) Delivery_time -> Predict delivery time using sorting time

# For reading data set
# importing necessary libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# Reading a csv file using pandas library
Delivery_time=pd.read_csv("delivery_time.csv")
Delivery_time.columns

Delivery_time = Delivery_time.rename(columns = {"Delivery Time":"DeliveryTime"})
Delivery_time = Delivery_time.rename(columns = {"Sorting Time":"SortingTime"})

plt.hist(Delivery_time.DeliveryTime)
plt.hist(Delivery_time.SortingTime)

plt.boxplot(Delivery_time.DeliveryTime)
plt.boxplot(Delivery_time.SortingTime)

plt.plot(Delivery_time.DeliveryTime,Delivery_time.SortingTime,"ro");plt.xlabel("Delivery Time");plt.ylabel("Sorting Time")

#Calories.corr()
Delivery_time.SortingTime.corr(Delivery_time.DeliveryTime) # # correlation value between X and Y cor(y,x)
np.corrcoef(Delivery_time.SortingTime,Delivery_time.DeliveryTime)

# For preparing linear regression model we need to import the statsmodels.formula.api
import statsmodels.formula.api as smf
model=smf.ols("SortingTime~DeliveryTime",data=Delivery_time).fit()

##type(model)
# For getting coefficients of the varibles used in equation
model.params 
# P-values for the variables and R-squared value for prepared model
model.summary()

model.conf_int(0.05) # 95% confidence interval

pred = model.predict(Delivery_time) # Predicted values of Sorting time using the model
pred
# Visualization of regresion line over the scatter plot of DeliveryTime and SortingTime
# For visualization we need to import matplotlib.pyplot
import matplotlib.pylab as plt
plt.scatter(x=Delivery_time['DeliveryTime'],y=Delivery_time['SortingTime'],color='red');plt.plot(Delivery_time['DeliveryTime'],pred,color='black');plt.xlabel('DeliveryTime');plt.ylabel('SortingTime')
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(Delivery_time.SortingTime,pred))
rmse

#pred.corr(Delivery_time.SortingTime) # 0.81
# Transforming variables for accuracy
model2 = smf.ols('SortingTime~np.log(DeliveryTime)',data=Delivery_time).fit()
model2.params
model2.summary()
print(model2.conf_int(0.01)) # 99% confidence level
pred2 = model2.predict(Delivery_time)
pred2.corr(Delivery_time.SortingTime)
# pred2 = model2.predict(Delivery_time.iloc[:,0])
pred2
plt.scatter(x=Delivery_time['DeliveryTime'],y=Delivery_time['SortingTime'],color='green');plt.plot(Delivery_time['DeliveryTime'],pred2,color='blue');plt.xlabel('DeliveryTime');plt.ylabel('SortingTime')

# Exponential transformation
model3 = smf.ols('np.log(SortingTime)~DeliveryTime',data=Delivery_time).fit()
model3.params
model3.summary()
print(model3.conf_int(0.01)) # 99% confidence level
pred_log = model3.predict(Delivery_time)
pred_log
pred3=np.exp(pred_log)  # as we have used log(SortingTime) in preparing model so we need to convert it back
pred3
pred3.corr(Delivery_time.SortingTime)
plt.scatter(x=Delivery_time['DeliveryTime'],y=Delivery_time['SortingTime'],color='green');plt.plot(Delivery_time.DeliveryTime,np.exp(pred_log),color='blue');plt.xlabel('Delivery Time');plt.ylabel('Sorting Time')
resid_3 = pred3-Delivery_time.SortingTime
# so we will consider the model having highest R-Squared value which is the log transformation - model3
# getting residuals of the entire data set
student_resid = model3.resid_pearson 
student_resid
plt.plot(pred3,model3.resid_pearson,"o");plt.axhline(y=0,color='green');plt.xlabel("Observation Number");plt.ylabel("Standardized Residual")

# Predicted vs actual values
plt.scatter(x=pred3,y=Delivery_time.SortingTime);plt.xlabel("Predicted");plt.ylabel("Actual")

# Quadratic model
Delivery_time["Sorting_Sq"] = Delivery_time.DeliveryTime*Delivery_time.DeliveryTime
Delivery_time.Sorting_Sq
model_quad = smf.ols("SortingTime~DeliveryTime+Delivery_time.Sorting_Sq",data=Delivery_time).fit()
model_quad.params
model_quad.summary()
pred_quad = model_quad.predict(Delivery_time)

model_quad.conf_int(0.05) 
plt.scatter(Delivery_time.DeliveryTimeTime,Delivery_time.SortingTime,c="b");plt.plot(Delivery_time.DeliveryTime,pred_quad,"r")

plt.hist(model_quad.resid_pearson) # histogram for residual values 