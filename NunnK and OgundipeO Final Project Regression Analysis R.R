# Kyle Nunn and Omoyeni M. Ogundipe
# Regression Analysis : Project
# Logistic Regression: Credit Card/ Loan Defaults

library(data.table)
setwd("~/Desktop/Regression Analysis")
default.data <- read.csv("Credit Card Default Data.csv")
attach(default.data)
dim(default.data)
default.data

# Code replicated from https://rpubs.com/SameerMathur/LR_CreditCardDefault_Taiwan
# First, we are going to run logistic regression based on the first couple of variables for simplicity sake, including:
# Afterwards, we will run a logistic regression model with all of the variables in the data set.
# After running the first and second models, we will run a logistic regression model on a new set of data and compare the two data sets
# After comparing the two data sets, we will make our conclusion as to which variables have the best correlation to predicting default loans
# After drawing conclusions, we will make a hypothesis as to which variables that were not included in each data set might be beneficial for predicting probability of default.

head(default.data)
str(default.data)

# Now we want to convert some of the variables that are integers into factors including : ID, SEX, EDUCATION, MARRIAGE, and DEFAULT
default.data$ID <- as.factor(default.data$ID)
default.data$SEX <- as.factor(default.data$SEX)
default.data$EDUCATION <- as.factor(default.data$EDUCATION)
default.data$MARRIAGE <- as.factor(default.data$MARRIAGE)
default.data$default.payment.next.month <- as.factor(default.data$default.payment.next.month)
str(default.data)

# Changing levels of Default variable
levels(default.data$default.payment.next.month) <- c("No","Yes")

# Varifying conversion
str(default.data)

# Now we are going to split the data into training and testing data
library(caret)
set.seed(2341)

trainIndex <- createDataPartition(default.data$default.payment.next.month, p = .8, list = FALSE)
traindata <- default.data[trainIndex,]
testdata <- default.data[-trainIndex,]
dim(traindata)
dim(testdata)

# Now we will train the logistic regression model
# First, we will set the control parameters using the bootstrap
objControl <- trainControl(method = "boot", 
                           number = 2, 
                           returnResamp = 'none', 
                           summaryFunction = twoClassSummary, 
                           classProbs = TRUE,
                           savePredictions = TRUE)

# Now we will run the training model
set.seed(766)
# model building using caret package
LRModel <- train(default.payment.next.month ~ LIMIT_BAL
                 + SEX
                 + EDUCATION
                 + MARRIAGE
                 + AGE, 
                 data = traindata,
                 method = 'glm',
                 trControl = objControl,
                 metric = "ROC")

# summary of the model
summary(LRModel)

# According the summary of the output of the model, it appears as if there are some statistically significant variables that will sufficiently predict the probability of defaulting next month.
# While it may appear that way, we know that while they are technically statistically significant, the estimates are so low that they are of no real value.

# predicting the model on test data set
PredLR <- predict(LRModel, testdata,type = "prob")

# Now we will predict the predicted probabilities
# plot of probabilities
plot(PredLR$Yes, 
     main = "Scatterplot of Probabilities of default (test data)", 
     xlab = "Customer ID", 
     ylab = "Predicted Probability of default")

# Obviously, using the above predictor variables are not good predictors of whether someone will default on their loan.
# Now we will locate the range of predicted probabilities
range <- range(PredLR$Yes)
format(range, scientific = FALSE)

# Now we will make confusion matrix cut-off probability at .20

pred.LR <- ifelse(PredLR$Yes > 0.20, "Yes", "No")
Predicted <- ordered(pred.LR, levels = c("Yes", "No"))

# actual and predicted data columns
Predicted <- as.factor(Predicted)
Actual <- as.factor(testdata$default.payment.next.month)
# making confusion matrix
cm <-confusionMatrix(data =Predicted,reference = Actual,
                     positive = "Yes")
cm

# Now we will calculate accuracy, sensitivity, and specificity. 

# function to print confusion matrices for diffrent cut-off levels of probability
CmFn <- function(cutoff) {
  
  # predicting the test set results
  Pred.LR <- predict(LRModel, testdata,type = "prob")
  C1 <- ifelse(Pred.LR$Yes > cutoff, "Yes", "No")
  C2 <- testdata$default.payment.next.month
  predY   <- as.factor(C1)
  actualY <- as.factor(C2)
  
  # ordering the levels of predicted variable
  predY <- ordered(predY, levels = c("Yes", "No"))
  
  # use the confusionMatrix from the caret package
  cm1 <-confusionMatrix(data = predY,reference = actualY, positive = "Yes")
  # extracting accuracy
  Accuracy <- cm1$overall[1]
  # extracting sensitivity
  Sensitivity <- cm1$byClass[1]
  # extracting specificity
  Specificity <- cm1$byClass[2]
  # extracting value of kappa
  Kappa <- cm1$overall[2]
  
  # combined table
  tab <- cbind(Accuracy,Sensitivity,Specificity,Kappa)
  return(tab)}
# sequence of cut-off probability       
cutoff1 <- seq( .01, .4, by = .03 )

# loop using "lapply"
tab2    <- lapply(cutoff1, CmFn)
# extra coding for saving table as desired format
tab3 <- rbind(tab2[[1]],tab2[[2]],tab2[[3]],tab2[[4]],tab2[[5]],tab2[[6]],tab2[[7]],
              tab2[[8]],tab2[[9]],tab2[[10]],tab2[[11]],tab2[[12]],tab2[[13]],tab2[[14]])
# printing the table
tab4 <- as.data.frame(tab3)
tab5 <- cbind(cutoff1,tab4$Accuracy,tab4$Sensitivity,tab4$Specificity,tab4$Kappa)
tab6 <- as.data.frame(tab5)

pm <- setnames(tab6, "cutoff1", "cutoff")
pm <- setnames(pm, "V2", "Accuracy")
pm <- setnames(pm, "V3", "Senstivity")
pm <- setnames(pm, "V4", "Specificity")
pm <- setnames(pm, "V5", "kappa")
pm

# Now we will plot ROC Curve
library(ROCR)

# Plotting the curves
PredLR <- predict(LRModel, testdata,type = "prob")
lgPredObj <- prediction(PredLR[2],testdata$default.payment.next.month)
lgPerfObj <- performance(lgPredObj, "tpr","fpr")
# plotting ROC curve
plot(lgPerfObj,main = "ROC Curve",col = 2,lwd = 2)
abline(a = 0,b = 1,lwd = 2,lty = 3,col = "black")

# Finding area under the curve
aucLR <- performance(lgPredObj, measure = "auc")
aucLR <- aucLR@y.values[[1]]
aucLR

# Now we move on to the full logistic regression model for the full data set

# Following R code borrowed from https://www.r-bloggers.com/2019/11/logistic-regression-in-r-a-classification-technique-to-predict-credit-card-default/

# First step, we will import the neccessary libraries (including ones from above)

library(knitr)
library(tidyverse)
library(ggplot2)
library(lattice)
library(reshape2)

# Now we will import the data set again

DefaultData <- read.csv("Credit Card Default Data.csv")
head(DefaultData1) 

# We will now rename "default payment next month" to simply "default_payment to avoid any discrepancies and to make things simpler.

colnames(DefaultData)[colnames(DefaultData)=="default payment next month"] <- "default_payment"
head(DefaultData)

# As we have learned throughout the course, we will now conduct exploratory data analaysis
# In doing so, we will be able to visualize the data, find relations between different variables, and even deal with missing values and outliers.

dim(DefaultData)
str(DefaultData)

DefaultData[, 1:25] <- sapply(DefaultData[, 1:25], as.character)
str(DefaultData)

DefaultData[, 1:25] <- sapply(DefaultData[, 1:25], as.numeric)
str(DefaultData)

summary(DefaultData)

# Finding how much of each categorical variable there is in the dataset:
# Also, we are going to 'attach' the data so we can use headings in code
attach(DefaultData)
count(DefaultData, vars = EDUCATION)
count(DefaultData, vars = MARRIAGE)

# Now, for simplicity sake, we are going to converge 'like' variables or ones we don't know that much about..
DefaultData$EDUCATION[DefaultData$EDUCATION == 0] <- 4
DefaultData$EDUCATION[DefaultData$EDUCATION == 5] <- 4
DefaultData$EDUCATION[DefaultData$EDUCATION == 6] <- 4
DefaultData$MARRIAGE[DefaultData$MARRIAGE == 0] <- 3
count(DefaultData, vars = MARRIAGE)
count(DefaultData, vars = EDUCATION)

# Now we can move on the multi-variate analysis of the variables in the data set, introducing a heat map.
install.packages("DataExplorer")
library(DataExplorer)


plot_correlation(na.omit(DefaultData), maxcat = 5L)

# We can see right off the bat that the factors such as Sex, Education, Marriage, and Age don't have a strong correlation with next month default probability.
# Looking at the heat map, we can kind of tell which predictor variables have an extremely low (we don't technically know they are signficantly low)
# Knowing which predictors have low correlation, we can delete them from the data frame to make our model clearer

DefaultData_New <- select(DefaultData, -one_of('ID','AGE', 'BILL_AMT2',
                                 'BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6'))
head(DefaultData_New)

# Now we need to standardize the data
# As we have learned, when we standardize, the mean of the data is now 0 and the standard deviation is 1

DefaultData_New[, 1:17] <- scale(DefaultData_New[, 1:17])
head(DefaultData_New)

# Now that we have standardized the data, we can split the data into a training and testing set
# We we use 70% of the data for training and 30% of the data for testing

data2 <- sort(sample(nrow(DefaultData_New), nrow(DefaultData_New)*.7))
train2 <- DefaultData_New[data2,]
test2 <- DefaultData_New[-data2,]
dim(train2)
dim(test2)

# Let's build our model now!! Logistic Regression is cool!

log.model <- glm(default.payment.next.month ~., data = train, family = binomial(link = "logit"))
summary(log.model)

test2[1:10,]

log.predictions <- predict(log.model, test2, type="response")
## Look at probability output
head(log.predictions, 10)

log.prediction.rd <- ifelse(log.predictions > 0.5, 1, 0)
head(log.prediction.rd, 10)

# Now we can evaluate the model using a confusion matrix to see what percentage of the time our model will correctly predict a default or no default.

table(log.prediction.rd, test2[,18])
log.prediction.rd
accuracy <- table(log.prediction.rd, test2[,18])
accuracy


sum(diag(accuracy))/sum(accuracy)






