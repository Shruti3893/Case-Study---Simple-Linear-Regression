install.packages("readr")
library(readr)
setwd("C://Users//Lenovo//Desktop//ExcelR//Assignments//Simple Linear Regression")
getwd()

Emp_data<-read.csv("emp_data.csv")

View(Emp_data)

#Exploratory Data Analysis

summary(Emp_data)

#Scatter plot

plot(Emp_data$Churn_out_rate,Emp_data$Salary_hike)

attach(Emp_data)

sum(is.na(Emp_data))

#Correlation Coefficient (r)
cor(Emp_data$Churn_out_rate,Emp_data$Salary_hike)

# Simple Linear Regression model
reg <- lm(Emp_data$Churn_out_rate ~ Emp_data$Salary_hike) # lm(Y ~ X)

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

# x = log(Emp_data$Salary_hike); y = Emp_data$Churn_out_rate

plot(log(Emp_data$Salary_hike), Emp_data$Churn_out_rate)
cor(log(Emp_data$Salary_hike), Emp_data$Churn_out_rate)

reg_log <- lm(Emp_data$Churn_out_rate ~ log(Emp_data$Salary_hike))   # lm(Y ~ X)

summary(reg_log)

predict(reg_log)

reg_log$residuals
sqrt(sum(reg_log$residuals^2)/nrow(Emp_data))  #RMSE

confint(reg_log,level=0.95)
predict(reg_log,interval="confidence")

######################

# Exponential Model

# x = Emp_data$Salary_hike and y = log(Emp_data$Churn_out_rate)

plot(Emp_data$Salary_hike, log(Emp_data$Churn_out_rate))

cor(Emp_data$Salary_hike, log(Emp_data$Churn_out_rate))

reg_exp <- lm(log(Emp_data$Churn_out_rate) ~ Emp_data$Salary_hike)  #lm(log(Y) ~ X)

summary(reg_exp)

reg_exp$residuals

sqrt(mean(reg_exp$residuals^2))

logchurn <- predict(reg_exp)
churn <- exp(logchurn)

error = Emp_data$Churn_out_rate - churn
error

sqrt(sum(error^2)/nrow(Emp_data))  #RMSE

confint(reg_exp,level=0.95)
predict(reg_exp,interval="confidence")
