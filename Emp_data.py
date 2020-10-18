# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 17:50:16 2020

@author: Lenovo
"""

#Problem Statement : Do the necessary transformations for input variables for getting 
#better R^2 value for the model prepared.
#3) Emp_data -> Build a prediction model for Churn_out_rate 

# For reading data set
# importing necessary libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# Reading a csv file using pandas library
Emp_data=pd.read_csv("emp_data.csv")
Emp_data.columns

plt.hist(Emp_data.Salary_hike)
plt.hist(Emp_data.Churn_out_rate)

plt.boxplot(Emp_data.Salary_hike)
plt.boxplot(Emp_data.Churn_out_rate)

plt.plot(Emp_data.Salary_hike,Emp_data.Churn_out_rate,"ro");plt.xlabel("Salary_hike");plt.ylabel("Churn_out_rate")

#Emp_data.corr()
Emp_data.Churn_out_rate.corr(Emp_data.Salary_hike) # # correlation value between X and Y cor(y,x)
np.corrcoef(Emp_data.Churn_out_rate,Emp_data.Salary_hike)

# For preparing linear regression model we need to import the statsmodels.formula.api
import statsmodels.formula.api as smf
model=smf.ols("Churn_out_rate~Salary_hike",data=Emp_data).fit()

##type(model)
# For getting coefficients of the varibles used in equation
model.params 
# P-values for the variables and R-squared value for prepared model
model.summary()

model.conf_int(0.05) # 95% confidence interval

pred = model.predict(Emp_data) # Predicted values of Churn_out_rate using the model
pred
# Visualization of regresion line over the scatter plot of Salary_hike and Churn_out_rate
# For visualization we need to import matplotlib.pyplot
import matplotlib.pylab as plt
plt.scatter(x=Emp_data['Salary_hike'],y=Emp_data['Churn_out_rate'],color='red');plt.plot(Emp_data['Salary_hike'],pred,color='black');plt.xlabel('Salary Hike');plt.ylabel('Churn out rate')
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(Emp_data.Churn_out_rate,pred))
rmse

#pred.corr(Emp_data.Churn_out_rate) # 0.81
# Transforming variables for accuracy
model2 = smf.ols('Churn_out_rate~np.log(Salary_hike)',data=Emp_data).fit()
model2.params
model2.summary()
print(model2.conf_int(0.01)) # 99% confidence level
pred2 = model2.predict(Emp_data)
pred2.corr(Emp_data.Churn_out_rate)
# pred2 = model2.predict(Emp_data.iloc[:,0])
pred2
plt.scatter(x=Emp_data['Salary_hike'],y=Emp_data['Churn_out_rate'],color='green');plt.plot(Emp_data['Salary_hike'],pred2,color='blue');plt.xlabel('Salary Hike');plt.ylabel('Churn out rate')

# Exponential transformation
model3 = smf.ols('np.log(Churn_out_rate)~Salary_hike',data=Emp_data).fit()
model3.params
model3.summary()
print(model3.conf_int(0.01)) # 99% confidence level
pred_log = model3.predict(Emp_data)
pred_log
pred3=np.exp(pred_log)  # as we have used log(Churn_out_rate) in preparing model so we need to convert it back
pred3
pred3.corr(Emp_data.Churn_out_rate)
plt.scatter(x=Emp_data['Salary_hike'],y=Emp_data['Churn_out_rate'],color='green');plt.plot(Emp_data.Salary_hike,np.exp(pred_log),color='blue');plt.xlabel('Salary Hike');plt.ylabel('Churn out rate')
resid_3 = pred3-Emp_data.Churn_out_rate
# so we will consider the model having highest R-Squared value which is the log transformation - model3
# getting residuals of the entire data set
student_resid = model3.resid_pearson 
student_resid
plt.plot(pred3,model3.resid_pearson,"o");plt.axhline(y=0,color='green');plt.xlabel("Observation Number");plt.ylabel("Standardized Residual")

# Predicted vs actual values
plt.scatter(x=pred3,y=Emp_data.Churn_out_rate);plt.xlabel("Predicted");plt.ylabel("Actual")

# Quadratic model
Emp_data["Weight_Sq"] = Emp_data.Salary_hike*Emp_data.Salary_hike
Emp_data.Weight_Sq
model_quad = smf.ols("Churn_out_rate~Salary_hike+Emp_data.Weight_Sq",data=Emp_data).fit()
model_quad.params
model_quad.summary()
pred_quad = model_quad.predict(Emp_data)

model_quad.conf_int(0.05) 
plt.scatter(Emp_data.Salary_hike,Emp_data.Churn_out_rate,c="b");plt.plot(Emp_data.Salary_hike,pred_quad,"r")

plt.hist(model_quad.resid_pearson) # histogram for residual values 