# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 17:50:16 2020

@author: Lenovo
"""

#Problem Statement : Do the necessary transformations for input variables for getting 
#better R^2 value for the model prepared.
#4) Salary_hike -> Build a prediction model for Salary_hike

# For reading data set
# importing necessary libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# Reading a csv file using pandas library
Salary_hike=pd.read_csv("salary_data.csv")
Salary_hike.columns

plt.hist(Salary_hike.YearsExperience)
plt.hist(Salary_hike.Salary)

plt.boxplot(Salary_hike.YearsExperience)
plt.boxplot(Salary_hike.Salary)

plt.plot(Salary_hike.YearsExperience,Salary_hike.Salary,"ro");plt.xlabel("Years Of Experience");plt.ylabel("Salary")

#Salary_hike.corr()
Salary_hike.Salary.corr(Salary_hike.YearsExperience) # # correlation value between X and Y cor(y,x)
np.corrcoef(Salary_hike.Salary,Salary_hike.YearsExperience)

# For preparing linear regression model we need to import the statsmodels.formula.api
import statsmodels.formula.api as smf
model=smf.ols("Salary~YearsExperience",data=Salary_hike).fit()

##type(model)
# For getting coefficients of the varibles used in equation
model.params 
# P-values for the variables and R-squared value for prepared model
model.summary()

model.conf_int(0.05) # 95% confidence interval

pred = model.predict(Salary_hike) # Predicted values of Salary using the model
pred
# Visualization of regresion line over the scatter plot of YearsExperience and Salary
# For visualization we need to import matplotlib.pyplot
import matplotlib.pylab as plt
plt.scatter(x=Salary_hike['YearsExperience'],y=Salary_hike['Salary'],color='red');plt.plot(Salary_hike['YearsExperience'],pred,color='black');plt.xlabel('Years Of Experience');plt.ylabel('Salary')
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(Salary_hike.Salary,pred))
rmse

#pred.corr(Salary_hike.Salary) # 0.81
# Transforming variables for accuracy
model2 = smf.ols('Salary~np.log(YearsExperience)',data=Salary_hike).fit()
model2.params
model2.summary()
print(model2.conf_int(0.01)) # 99% confidence level
pred2 = model2.predict(Salary_hike)
pred2.corr(Salary_hike.Salary)
# pred2 = model2.predict(Salary_hike.iloc[:,0])
pred2
plt.scatter(x=Salary_hike['YearsExperience'],y=Salary_hike['Salary'],color='green');plt.plot(Salary_hike['YearsExperience'],pred2,color='blue');plt.xlabel('Years Of Experience');plt.ylabel('Salary')

# Exponential transformation
model3 = smf.ols('np.log(Salary)~YearsExperience',data=Salary_hike).fit()
model3.params
model3.summary()
print(model3.conf_int(0.01)) # 99% confidence level
pred_log = model3.predict(Salary_hike)
pred_log
pred3=np.exp(pred_log)  # as we have used log(Salary) in preparing model so we need to convert it back
pred3
pred3.corr(Salary_hike.Salary)
plt.scatter(x=Salary_hike['YearsExperience'],y=Salary_hike['Salary'],color='green');plt.plot(Salary_hike.YearsExperience,np.exp(pred_log),color='blue');plt.xlabel('Years Of Experience');plt.ylabel('Salary')
resid_3 = pred3-Salary_hike.Salary
# so we will consider the model having highest R-Squared value which is the log transformation - model3
# getting residuals of the entire data set
student_resid = model3.resid_pearson 
student_resid
plt.plot(pred3,model3.resid_pearson,"o");plt.axhline(y=0,color='green');plt.xlabel("Observation Number");plt.ylabel("Standardized Residual")

# Predicted vs actual values
plt.scatter(x=pred3,y=Salary_hike.Salary);plt.xlabel("Predicted");plt.ylabel("Actual")

# Quadratic model
Salary_hike["Weight_Sq"] = Salary_hike.YearsExperience*Salary_hike.YearsExperience
Salary_hike.Weight_Sq
model_quad = smf.ols("Salary~YearsExperience+Salary_hike.Weight_Sq",data=Salary_hike).fit()
model_quad.params
model_quad.summary()
pred_quad = model_quad.predict(Salary_hike)

model_quad.conf_int(0.05) 
plt.scatter(Salary_hike.YearsExperience,Salary_hike.Salary,c="b");plt.plot(Salary_hike.YearsExperience,pred_quad,"r")

plt.hist(model_quad.resid_pearson) # histogram for residual values 