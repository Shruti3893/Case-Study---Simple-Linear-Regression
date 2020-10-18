install.packages("readr")
library(readr)
setwd("C://Users//Lenovo//Desktop//ExcelR//Assignments//Simple Linear Regression")
getwd()

Calories_consumed<-read.csv("calories_consumed.csv")
View(Calories_consumed)

#Exploratory Data Analysis

summary(Calories_consumed)

#Scatter plot

plot(Calories_consumed$Weight.gained..grams.,Calories_consumed$Calories.Consumed)

attach(Calories_consumed)
sum(is.na(Calories_consumed))

#Correlation Coefficient (r)
cor(Calories_consumed$Weight.gained..grams.,Calories_consumed$Calories.Consumed)

# Simple Linear Regression model
reg <- lm(Calories_consumed$Weight.gained..grams. ~ Calories_consumed$Calories.Consumed) # lm(Y ~ X)

summary(reg)

pred <- predict(reg)
reg$residuals
sum(reg$residuals)
mean(reg$residuals)
sqrt(sum(reg$residuals^2)/nrow(Calories_consumed))  #RMSE
sqrt(mean(reg$residuals^2))
confint(reg,level=0.95)
predict(reg,interval="predict")

# Logarithmic Model

# x = log(Calories.Consumed); y = Weight.gained..grams.

plot(log(Calories_consumed$Calories.Consumed), Calories_consumed$Weight.gained..grams.)
cor(log(Calories_consumed$Calories.Consumed), Calories_consumed$Weight.gained..grams.)

reg_log <- lm(Calories_consumed$Weight.gained..grams. ~ log(Calories_consumed$Calories.Consumed))   # lm(Y ~ X)

summary(reg_log)

predict(reg_log)

reg_log$residuals
sqrt(sum(reg_log$residuals^2)/nrow(Calories_consumed))  #RMSE

confint(reg_log,level=0.95)
predict(reg_log,interval="confidence")

######################

# Exponential Model

# x = Calories.Consumed and y = log(Weight.gained..grams.)

plot(Calories_consumed$Calories.Consumed, log(Calories_consumed$Weight.gained..grams.))

cor(Calories_consumed$Calories.Consumed, log(Calories_consumed$Weight.gained..grams.))

reg_exp <- lm(log(Calories_consumed$Weight.gained..grams.) ~ Calories_consumed$Calories.Consumed)  #lm(log(Y) ~ X)

summary(reg_exp)
reg_exp$residuals
sqrt(mean(reg_exp$residuals^2))

logweight <- predict(reg_exp)
weight <- exp(logweight)

error = Calories_consumed$Weight.gained..grams. - weight
error

sqrt(sum(error^2)/nrow(Calories_consumed))  #RMSE

confint(reg_exp,level=0.95)
predict(reg_exp,interval="confidence")