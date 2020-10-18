install.packages("readr")
library(readr)
setwd("C://Users//Lenovo//Desktop//ExcelR//Assignments//Simple Linear Regression")
getwd()

Emp_data<-read.csv("emp_data.csv")

View(Emp_data)

#Exploratory Data Analysis

summary(Emp_data)

#Scatter plot

plot(Emp_data$Salary_hike,Emp_data$Churn_out_rate)

attach(Emp_data)

sum(is.na(Emp_data))

#Correlation Coefficient (r)
cor(Emp_data$Salary_hike,Emp_data$Churn_out_rate)

# Simple Linear Regression model
reg <- lm(Emp_data$Salary_hike ~ Emp_data$Churn_out_rate) # lm(Y ~ X)

summary(reg)

pred <- predict(reg)

reg$residuals
sum(reg$residuals)
mean(reg$residuals)
sqrt(sum(reg$residuals^2)/nrow(Emp_data))  #RMSE
sqrt(mean(reg$residuals^2))
confint(reg,level=0.95)
predict(reg,interval="predict")

# Logrithamic Model

# x = log(Emp_data$Churn_out_rate); y = Emp_data$Salary_hike

plot(log(Emp_data$Churn_out_rate), Emp_data$Salary_hike)
cor(log(Emp_data$Churn_out_rate), Emp_data$Salary_hike)

reg_log <- lm(Emp_data$Salary_hike ~ log(Emp_data$Churn_out_rate))   # lm(Y ~ X)

summary(reg_log)

predict(reg_log)

reg_log$residuals
sqrt(sum(reg_log$residuals^2)/nrow(Emp_data))  #RMSE

confint(reg_log,level=0.95)
predict(reg_log,interval="confidence")

######################

# Exponential Model

# x = Emp_data$Churn_out_rate and y = log(Emp_data$Salary_hike)

plot(Emp_data$Churn_out_rate, log(Emp_data$Salary_hike))

cor(Emp_data$Churn_out_rate, log(Emp_data$Salary_hike))

reg_exp <- lm(log(Emp_data$Salary_hike) ~ Emp_data$Churn_out_rate)  #lm(log(Y) ~ X)

summary(reg_exp)

reg_exp$residuals

sqrt(mean(reg_exp$residuals^2))

logsalaryhike <- predict(reg_exp)
salaryhike <- exp(logsalaryhike)

error = Emp_data$Salary_hike - salaryhike
error

sqrt(sum(error^2)/nrow(Emp_data))  #RMSE

confint(reg_exp,level=0.95)
predict(reg_exp,interval="confidence")