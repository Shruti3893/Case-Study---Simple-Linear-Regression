# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 17:50:16 2020

@author: Lenovo
"""

#Problem Statement : Do the necessary transformations for input variables for getting 
#better R^2 value for the model prepared.
#1) Calories_consumed-> predict weight gained using calories consumed

# For reading data set
# importing necessary libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# Reading a csv file using pandas library
Calories=pd.read_csv("calories_consumed.csv")
Calories.columns

Calories = Calories.rename(columns = {"Weight gained (grams)":"WeightGained"})
Calories = Calories.rename(columns = {"Calories Consumed":"CaloriesConsumed"})

plt.hist(Calories.WeightGained)
plt.hist(Calories.CaloriesConsumed)

plt.boxplot(Calories.WeightGained)
plt.boxplot(Calories.CaloriesConsumed)

plt.plot(Calories.WeightGained,Calories.CaloriesConsumed,"ro");plt.xlabel("Weight Gained");plt.ylabel("Calories Consumed")

#Calories.corr()
Calories.CaloriesConsumed.corr(Calories.WeightGained) # # correlation value between X and Y cor(y,x)
np.corrcoef(Calories.CaloriesConsumed,Calories.WeightGained)

# For preparing linear regression model we need to import the statsmodels.formula.api
import statsmodels.formula.api as smf
model=smf.ols("CaloriesConsumed~WeightGained",data=Calories).fit()

##type(model)
# For getting coefficients of the varibles used in equation
model.params 
# P-values for the variables and R-squared value for prepared model
model.summary()

model.conf_int(0.05) # 95% confidence interval

pred = model.predict(Calories) # Predicted values of Calories Consumed using the model
pred
# Visualization of regresion line over the scatter plot of WeightGained and CaloriesConsumed
# For visualization we need to import matplotlib.pyplot
import matplotlib.pylab as plt
plt.scatter(x=Calories['WeightGained'],y=Calories['CaloriesConsumed'],color='red');plt.plot(Calories['WeightGained'],pred,color='black');plt.xlabel('WeightGained');plt.ylabel('CaloriesConsumed')
from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(Calories.CaloriesConsumed,pred))
rmse

#pred.corr(Calories.CaloriesConsumed) # 0.81
# Transforming variables for accuracy
model2 = smf.ols('CaloriesConsumed~np.log(WeightGained)',data=Calories).fit()
model2.params
model2.summary()
print(model2.conf_int(0.01)) # 99% confidence level
pred2 = model2.predict(Calories)
pred2.corr(Calories.CaloriesConsumed)
# pred2 = model2.predict(Calories.iloc[:,0])
pred2
plt.scatter(x=Calories['WeightGained'],y=Calories['CaloriesConsumed'],color='green');plt.plot(Calories['WeightGained'],pred2,color='blue');plt.xlabel('WeightGained');plt.ylabel('CaloriesConsumed')

# Exponential transformation
model3 = smf.ols('np.log(CaloriesConsumed)~WeightGained',data=Calories).fit()
model3.params
model3.summary()
print(model3.conf_int(0.01)) # 99% confidence level
pred_log = model3.predict(Calories)
pred_log
pred3=np.exp(pred_log)  # as we have used log(CaloriesConsumed) in preparing model so we need to convert it back
pred3
pred3.corr(Calories.CaloriesConsumed)
plt.scatter(x=Calories['WeightGained'],y=Calories['CaloriesConsumed'],color='green');plt.plot(Calories.WeightGained,np.exp(pred_log),color='blue');plt.xlabel('WEIGHT');plt.ylabel('CALORIES')
resid_3 = pred3-Calories.CaloriesConsumed
# so we will consider the model having highest R-Squared value which is the log transformation - model3
# getting residuals of the entire data set
student_resid = model3.resid_pearson 
student_resid
plt.plot(pred3,model3.resid_pearson,"o");plt.axhline(y=0,color='green');plt.xlabel("Observation Number");plt.ylabel("Standardized Residual")

# Predicted vs actual values
plt.scatter(x=pred3,y=Calories.CaloriesConsumed);plt.xlabel("Predicted");plt.ylabel("Actual")

# Quadratic model
Calories["Weight_Sq"] = Calories.WeightGained*Calories.WeightGained
Calories.Weight_Sq
model_quad = smf.ols("CaloriesConsumed~WeightGained+Calories.Weight_Sq",data=Calories).fit()
model_quad.params
model_quad.summary()
pred_quad = model_quad.predict(Calories)

model_quad.conf_int(0.05) 
plt.scatter(Calories.WeightGained,Calories.CaloriesConsumed,c="b");plt.plot(Calories.WeightGained,pred_quad,"r")

plt.hist(model_quad.resid_pearson) # histogram for residual values 