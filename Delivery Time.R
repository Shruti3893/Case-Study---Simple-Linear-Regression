install.packages("readr")
library(readr)
setwd("C://Users//Lenovo//Desktop//ExcelR//Assignments//Simple Linear Regression")
getwd()

Delivery_time<-read.csv("delivery_time.csv")

View(Delivery_time)

#Exploratory Data Analysis

summary(Delivery_time)

#Scatter plot

plot(Delivery_time$Delivery.Time,Delivery_time$Sorting.Time)

attach(Delivery_time)

sum(is.na(Delivery_time))

#Correlation Coefficient (r)
cor(Delivery_time$Delivery.Time,Delivery_time$Sorting.Time)

# Simple Linear Regression model
reg <- lm(Delivery_time$Delivery.Time ~ Delivery_time$Sorting.Time) # lm(Y ~ X)

summary(reg)

pred <- predict(reg)

reg$residuals
sum(reg$residuals)
mean(reg$residuals)
sqrt(sum(reg$residuals^2)/nrow(Delivery_time))  #RMSE
sqrt(mean(reg$residuals^2))
confint(reg,level=0.95)
predict(reg,interval="predict")

# Logrithamic Model

# x = log(Delivery_time$Sorting.Time); y = Delivery_time$Delivery.Time

plot(log(Delivery_time$Sorting.Time), Delivery_time$Delivery.Time)
cor(log(Delivery_time$Sorting.Time), Delivery_time$Delivery.Time)

reg_log <- lm(Delivery_time$Delivery.Time ~ log(Delivery_time$Sorting.Time))   # lm(Y ~ X)

summary(reg_log)

predict(reg_log)

reg_log$residuals
sqrt(sum(reg_log$residuals^2)/nrow(Delivery_time))  #RMSE

confint(reg_log,level=0.95)
predict(reg_log,interval="confidence")

######################

# Exponential Model

# x = Delivery_time$Sorting.Time and y = log(Delivery_time$Delivery.Time)

plot(Delivery_time$Sorting.Time, log(Delivery_time$Delivery.Time))

cor(Delivery_time$Sorting.Time, log(Delivery_time$Delivery.Time))

reg_exp <- lm(log(Delivery_time$Delivery.Time) ~ Delivery_time$Sorting.Time)  #lm(log(Y) ~ X)

summary(reg_exp)

reg_exp$residuals

sqrt(mean(reg_exp$residuals^2))

logsorting <- predict(reg_exp)
sorting <- exp(logsorting)

error = Delivery_time$Delivery.Time - sorting
error

sqrt(sum(error^2)/nrow(Delivery_time))  #RMSE

confint(reg_exp,level=0.95)
predict(reg_exp,interval="confidence")
