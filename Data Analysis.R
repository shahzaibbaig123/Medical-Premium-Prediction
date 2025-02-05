# Installing and adding required libraries
install.packages("dplyr")
install.packages("janitor")
install.packages("tidyverse")
install.packages("ggplot2")
install.packages("readr")
install.packages("gridExtra")
install.packages("rsample")
library(dplyr)
library(tidyverse)
library(rsample)
library(randomForest)
library(janitor)
library(rpart)
library(rpart.plot)
library(gbm)
library(glmnet)
library(tree)
library(RColorBrewer)
library(gridExtra)
library(ggplot2)
library(pls)
library(jtools)

# Loading dataset
Medicalpremium <- read_csv("path to /Medicalpremium.csv")

# Getting the summary and view of the data
glimpse(Medicalpremium)
summary(Medicalpremium)

# Checking for missing values 
colSums(is.na(Medicalpremium))

# Data Cleaning 
Medicalpremium <- clean_names(Medicalpremium)

Medicalpremium <- Medicalpremium %>%
  mutate(diabetes = as.factor(case_when(diabetes == 0 ~ "No",
                                        diabetes == 1 ~ "Yes"))) %>%
  mutate(blood_pressure_problems = as.factor(case_when(blood_pressure_problems == 0 ~ "No",
                                                       blood_pressure_problems == 1 ~ "Yes"))) %>%
  mutate(any_transplants = as.factor(case_when(any_transplants == 0 ~ "No",
                                               any_transplants == 1 ~ "Yes"))) %>%
  mutate(any_chronic_diseases = as.factor(case_when(any_chronic_diseases == 0 ~ "No",
                                                    any_chronic_diseases == 1 ~ "Yes"))) %>%
  mutate(known_allergies = as.factor(case_when(known_allergies == 0 ~ "No",
                                               known_allergies == 1 ~ "Yes"))) %>%
  mutate(history_of_cancer_in_family = as.factor(case_when(history_of_cancer_in_family == 0 ~ "No",
                                                           history_of_cancer_in_family == 1 ~ "Yes")))

# Data Exploration
hist(Medicalpremium$premium_price)

v1 <- ggplot(Medicalpremium) +
  geom_boxplot(aes(y=premium_price, x=diabetes, fill=diabetes), show.legend=FALSE) +
  xlab("Diabetes") +
  ylab("Premium Price")

v2 <- ggplot(Medicalpremium) +
  geom_boxplot(aes(y=premium_price, x=any_transplants, fill=any_transplants), show.legend=FALSE) +
  xlab("Any Transplants") +
  ylab("Premium Price")

v3 <- ggplot(Medicalpremium) +
  geom_boxplot(aes(y=premium_price, x=any_chronic_diseases, fill=any_chronic_diseases), show.legend = FALSE) +
  xlab("Chronic Diseases") +
  ylab("Premium Price")

v4 <- ggplot(Medicalpremium) +
  geom_boxplot(aes(y=premium_price, x=blood_pressure_problems, fill=blood_pressure_problems), show.legend = FALSE) +
  xlab("Blood Pressure Problems") +
  ylab("Premium Price")

v5 <- ggplot(Medicalpremium) +
  geom_boxplot(aes(y=premium_price, x=known_allergies, fill=known_allergies), show.legend = FALSE) +
  xlab("Known Allergies") +
  ylab("Premium Price")

v6 <- ggplot(Medicalpremium) +
  geom_boxplot(aes(y=premium_price, x=history_of_cancer_in_family, 
                   fill=history_of_cancer_in_family), show.legend = FALSE) +
  xlab("Cancer in Family") +
  ylab("Premium Price")

grid.arrange(v1, v2, v3, v4, v5, v6, nrow=2)




v7 <- ggplot(Medicalpremium) +
  geom_point(aes(x=age,y=premium_price)) +
  geom_smooth(aes(x=age,y=premium_price)) +
  xlab("Age (years)") +
  ylab("Premium Price")

v8 <- ggplot(Medicalpremium) +
  geom_point(aes(x=weight,y=premium_price)) +
  geom_smooth(aes(x=weight,y=premium_price), colour="green") +
  xlab("Weight (kg)") +
  ylab("Premium Price")

v9 <- ggplot(Medicalpremium) +
  geom_point(aes(x=height,y=premium_price)) +
  geom_smooth(aes(x=height,y=premium_price), colour="red") +
  xlab("Height (cm)") +
  ylab("Premium Price")

v10 <- ggplot(Medicalpremium, mapping=aes(x=premium_price, y=factor(number_of_major_surgeries), 
                                          fill=factor(number_of_major_surgeries))) +
  geom_violin(color="red", fill="orange", alpha=0.2, show.legend = FALSE) +
  labs(fill="Number of Major Surgeries") +
  ylab("Number of Major Surgeries") +
  xlab("Premium Price")

grid.arrange(v7, v8, v9, v10, nrow=2)


# Splitting Data into testing and training sets
set.seed(11)
med.split <- initial_split(Medicalpremium, prop = 3/4)
med.train <- training(med.split)
med.test <- testing(med.split)


# Predictive Modeling
rsquared <- function(pred){
  if (length(pred)==length(med.test$premium_price)){
    r2 = 1 - (sum((med.test$premium_price-pred)^2)/sum((med.test$premium_price-mean(med.test$premium_price))^2))
  }
  if (length(pred)==length(med.train$premium_price)){
    r2 = 1 - (sum((med.train$premium_price-pred)^2)/sum((med.train$premium_price-mean(med.train$premium_price))^2))
  }
  return (r2)
}


MSE <- function(pred){
  if (length(pred)==length(med.test$premium_price)){
    mse = sum((med.test$premium_price-pred)^2)/length(med.test$premium_price)
  }
  if (length(pred)==length(med.train$premium_price)){
    mse = sum((med.train$premium_price-pred)^2)/length(med.train$premium_price)
  }
  return (mse)
}

# Using Linear Stepwise Regression

# Forward Selection
linear.fwd <- step(lm(premium_price ~., data=med.train), direction = c("forward"))

fwd.pred.train = predict(linear.fwd, med.train)
mse.fwd.train = MSE(fwd.pred.train)
r2.fwd.train = rsquared(fwd.pred.train)

fwd.pred.test = predict(linear.fwd, med.test)
mse.fwd.test = MSE(fwd.pred.test)
r2.fwd.test = rsquared(fwd.pred.test)

summary(linear.fwd)

# Backward Selection

linear.bwd = step(lm(premium_price ~., data=med.train), direction = c("backward"))

bwd.pred.train = predict(linear.bwd, med.train)
mse.bwd.train = MSE(bwd.pred.train)
r2.bwd.train = rsquared(bwd.pred.train)

bwd.pred.test = predict(linear.bwd, med.test)
mse.bwd.test = MSE(bwd.pred.test)
r2.bwd.test = rsquared(bwd.pred.test)

summary(linear.bwd)





# Regression Trees

#Standard Regression Tree

medcost.model <- rpart(premium_price ~., data = med.train, method = "anova")
rpart.plot(medcost.model, main = "Prediction of Yearly Medical Coverage Costs", 
           extra = 101, digits = -1, yesno = 2, type = 5)

pred.tree.train <- predict(medcost.model, med.train)
mse.tree.train <- MSE(pred.tree.train)
r2.tree.train <- rsquared(pred.tree.train)

pred.tree.test <- predict(medcost.model, med.test)
mse.tree.test <- MSE(pred.tree.test)
r2.tree.test <- rsquared(pred.tree.test)

# Random Forest

set.seed(11)
medcost.rf.model <- randomForest(premium_price ~., data = med.train, mtry = 3, importance = TRUE)

pred.rf.train <- predict(medcost.rf.model, med.train)
mse.rf.train <- MSE(pred.rf.train)
r2.rf.train <- rsquared(pred.rf.train)

pred.rf.test <- predict(medcost.rf.model, med.test)
mse.rf.test <- MSE(pred.rf.test)
r2.rf.test <- rsquared(pred.rf.test)

imp <- data.frame(importance(medcost.rf.model, type =1))
imp <- rownames_to_column(imp, var = "variable")
ggplot(imp, aes(x=reorder(variable, X.IncMSE), y=X.IncMSE, color=reorder(variable, X.IncMSE))) +
  geom_point(show.legend=FALSE, size=3) +
  geom_segment(aes(x=variable, xend=variable, y=0, yend=X.IncMSE), size=3, show.legend=FALSE) +
  xlab("") +
  ylab("% Increase in MSE") +
  labs(title = "Variable Importance for Prediction of Premium Price") +
  coord_flip() +
  scale_color_manual(values = colorRampPalette(brewer.pal(1,"Purples"))(10)) +
  theme_classic()

# Results on Test Set

p1 <- ggplot(mapping = aes(x = med.test$premium_price, y = pred.rf.test)) +
  geom_point(color = "#BF87B3") +
  geom_abline(slope = 1) +
  xlab("Predicted Premium Price") +
  ylab("Actual Premium Price") +
  labs(title = "Random Forest")


p2 <- ggplot(mapping = aes(x = med.test$premium_price, y = pred.tree.test)) +
  geom_point(color = "#000080") +
  geom_abline(slope = 1) +
  xlab("Predicted Premium Price") +
  ylab("Actual Premium Price") +
  labs(title = "Standard Regression Tree")

p3 <- ggplot(mapping = aes(x = med.test$premium_price, y = fwd.pred.test)) +
  geom_point(color = "turquoise") +
  geom_abline(slope = 1) +
  xlab("Predicted Premium Price") +
  ylab("Actual Premium Price") +
  labs(title = "Forward Selection")

p4 <- ggplot(mapping = aes(x = med.test$premium_price, y = bwd.pred.test)) +
  geom_point(color = "purple") +
  geom_abline(slope = 1) +
  xlab("Predicted Premium Price") +
  ylab("Actual Premium Price") +
  labs(title = "Backward Selection")



grid.arrange(p1, p2, p3, p4, nrow=2)



mse.df <- data.frame(rbind(c("Forward Selection", mse.fwd.test), 
                           c("Backward Selection", mse.bwd.test), 
                           c("Single Regression Tree", mse.tree.test), 
                           c("Random Forest", mse.rf.test)))

mse.df <- mse.df %>%
  mutate(X2 = as.numeric(X2)) %>%
  arrange(-desc(X2)) %>%
  rename("Regression Method" = X1, "Mean Squared Error" = X2)

mse.df

ggplot(mse.df, aes(x=reorder(`Regression Method`, `Mean Squared Error`), 
                   y=`Mean Squared Error`, fill=`Regression Method`)) +
  geom_col() +
  xlab("Regression Method") +
  theme(axis.text.x = element_blank()) +
  scale_fill_discrete(limits = mse.df$`Regression Method`) +
  labs(title = "Comparison of MSE Across All 4 Models")

