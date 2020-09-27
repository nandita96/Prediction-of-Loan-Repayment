library(gmodels)
library(lubridate)
library(plyr)
library(ggplot2)
library(caTools)
library(e1071)
library(ROCR)
library(caret)
library(ROSE)
library(Rcpp)
library(dplyr)

#Setting the working directory
setwd("E:\\Semester 1\\Data Mining\\Project\\Loan Risk")

# Loan Status Dataset
LoanStatus = read.csv("LoanStatsV1.csv")

#Data Exploration
table <- tbl_df(LoanStatus)
#head(table) #144 columns remaining



# First remove the columns that more than 50% of values are NaNs
df <- table[, colSums(is.na(table)) < length(table)/2]
str(df)


#checking the density of loan amont
summary(df$loan_amnt)
library(ggplot2)
plot<- ggplot(data=df, aes(loan_amnt)) + geom_density(fill="cyan4")
plot
summary(df$loan_amnt)


Loan_stat_df<-subset(df, ! is.na(df$loan_status))%>%
  group_by(loan_status)%>%
  dplyr::summarise(Number=n())
ggplot(data=Loan_stat_df, aes(x=loan_status, y=Number))+
  geom_bar(stat="identity",fill="steelblue")+
  geom_text(aes(label=Number),vjust=-0.3,size=3.5)+
  theme_minimal()+coord_flip()



#Selecting relevant features for model
features <- c("loan_status", "grade", "sub_grade", "open_acc","pub_rec", "dti", "delinq_2yrs",
              "inq_last_6mths", "annual_inc", "home_ownership",  "purpose", "addr_state",
              "loan_amnt","int_rate", "installment", "issue_d", "revol_bal", "revol_util")

raw.data <- subset(df, select = features) 
str(raw.data)


#Deleting empty rows
raw.data <- raw.data[!apply(raw.data == "", 1, all),]


#Merging the Targeted Columns
table(raw.data$loan_status)
raw.data = raw.data %>%
  filter(loan_status == "Charged Off" | loan_status == "Fully Paid")
table(raw.data$loan_status)

#Moreover, I am going to encode this variable so that 
#1 represents Fully Paid, and 
#0 represents Charged Off

#Preparing data for analysis
#int_rate variable
class(raw.data$int_rate) #It is factor, should be numeric
raw.data$int_rate <- as.numeric(sub("%","",raw.data$int_rate)) #Taking out % sign and converting into numeric
raw.data$int_rate <- raw.data$int_rate / 100
is.numeric(raw.data$int_rate) # TRUE
anyNA(raw.data$int_rate) #No missing values

#revol_util variable
class(raw.data$revol_util) #It is factor, should be numeric
raw.data$revol_util <- as.numeric(sub("%","",raw.data$revol_util)) #Taking out % sign and converting into numeric
raw.data$revol_util <- raw.data$revol_util / 100
is.numeric(raw.data$revol_util) # TRUE
anyNA(raw.data$revol_util) #There are missing values
index.NA <- which(is.na(raw.data$revol_util)) #766 missing values
raw.data$revol_util[index.NA] <- median(raw.data$revol_util, na.rm = TRUE) #All missing values replaced by median 0.497
anyNA(raw.data$revol_util) #No missing values

#revol_bal variable
class(raw.data$revol_bal) #It is factor, should be numeric
raw.data$revol_bal <- as.character(raw.data$revol_bal) #Converting into character
raw.data$revol_bal <- as.numeric(raw.data$revol_bal) # Converting into numeric
anyNA(raw.data$revol_bal) #No missing values

#installment variable
class(raw.data$installment) #It is factor, should be numeric
raw.data$installment <- as.character(raw.data$installment) #Converting into character
raw.data$installment <- as.numeric(raw.data$installment) #Converting into numeric
is.numeric(raw.data$installment) # TRUE
anyNA(raw.data$installment) #No missing values

#loan_amnt
class(raw.data$loan_amnt) #It is factor, should be numeric
raw.data$loan_amnt <- as.character(raw.data$loan_amnt) #Converting into character
raw.data$loan_amnt <- as.numeric(raw.data$loan_amnt) #Converting into numeric
is.numeric(raw.data$loan_amnt) # TRUE
anyNA(raw.data$loan_amnt) #No missing values

#annual_inc
class(raw.data$annual_inc) #It is factor, should be numeric
raw.data$annual_inc <- as.character(raw.data$annual_inc) #Converting into character
raw.data$annual_inc <- as.numeric(raw.data$annual_inc) #Converting into numeric
is.numeric(raw.data$annual_inc) # TRUE

#loan_status
class(raw.data$loan_status) #It is factor
raw.data$loan_status <- as.character(raw.data$loan_status)
is.character(raw.data$loan_status)
#Taking only rows where laon_status is fully paid or charged off
arg <- raw.data$loan_status=="Fully Paid" | raw.data$loan_status=="Charged Off"
raw.data <- subset(raw.data, arg==TRUE) #Number of observations reduced to 39786

#Encoding loan_status 0 - Charged Off, 1 - Fully paid
raw.data$loan_status <- ifelse(raw.data$loan_status=="Fully Paid",1,0)
raw.data$loan_status <- as.integer(raw.data$loan_status) #Converting to integer
is.integer(raw.data$loan_status)
anyNA(raw.data$loan_status)

#Turning loan_status to factor
raw.data$loan_status <- factor(raw.data$loan_status)

#MODEL BUILDING

#The best way to split our data into training and test set is by cross-validation method (exhaustive cross-validation or non-exhaustive cross-validation) that uses multiple test sets. However, cross-validation method is very time consuming process. For this reason, in this project, I have decided to use single test set method where I split dataset into training set (70%) and test set (30%) using sample.split function.

#Building the logistic regression model is straight forward in R. I have used glm function to make a classifier and provided the training set to learn the classifier. Then, I used predict function and test set to get prediction.

#str(raw.data)
loan.model <- raw.data 
anyNA(raw.data) # No missing values
dim(loan.model) #14 features + 1 response, 552,625 observations

# Splitting data set into training and test set
set.seed(123) #making results reproduciable

sample <- sample.split(loan.model$loan_status, 0.7)
train.data <- subset(loan.model, sample==TRUE)
test.data <- subset(loan.model, sample==FALSE)

#LOGISTIC REGRESSION
logistic.regressor <- glm(loan_status ~ ., family = "binomial", data = train.data)
#summary(logistic.regressor)

# Selected variables based on *** from summary(logistic.regressor)
logistic.regressor <- glm(loan_status ~ revol_util+revol_bal+installment+int_rate+loan_amnt+home_ownership+open_acc+pub_rec+pub_rec+annual_inc+pub_rec+inq_last_6mths+dti+delinq_2yrs , family = "binomial", data = train.data)

#Predicting outcomes on test data
prob_pred <- predict(logistic.regressor, newdata = test.data[-1], type = "response")

#classify them into two classes 1 or 0
y_pred = ifelse(prob_pred > 0.5, 1, 0)
#str(y_pred)
table(ActualValue = test.data$loan_status, PredictedValue= y_pred)

# Accuracy
print((27+10206)/(29+1674+27+10206)) # 85%

#######################evalution###########

#ROC

res <- predict(logistic.regressor, newdata = train.data, type = "response")
library(ROCR)

ROCRPred = prediction(res, train.data$loan_status)
ROCRPref = performance(ROCRPred,"tpr","fpr")

#tpr - True positive rate
#fpr - False positive rate

#plot(ROCRPref,colorize = TRUE,print.cutoffs.at = seq(0.1,by = 0.1))

auc <- performance(ROCRPred, measure = "auc")
auc <- auc@y.values[[1]]
auc 




#NAIVE BAYES
# Fitting NAIVE BAYES to the Training set
library(e1071)
classifier = naiveBayes(x = train.data[-1],
                        y = train.data$loan_status) 

# Predicting the Test set results
y_pred = predict(classifier, newdata = test.data[-1])

# Making the Confusion Matrix
table(test.data$loan_status, y_pred)

# Accuracy
print((562+8513)/(562+8513+1722+1139)) # 76%

#For Random FOrest it cannot handle more than 53 factors. So removed one column
str(loan.model)
loan.model$issue_d = NULL
#set.seed(123) #making results reproduciable

sample <- sample.split(loan.model$loan_status, 0.7)
train.data <- subset(loan.model, sample==TRUE)
test.data <- subset(loan.model, sample==FALSE)

#RANDOM FOREST
library(randomForest)
set.seed(123)
classifier = randomForest(x = train.data[-1],y = train.data$loan_status)

# Predicting the Test set results
y_pred = predict(classifier, newdata = test.data[-1])

# Making the Confusion Matrix
table(test.data$loan_status, y_pred)

# Accuracy
print((1441+3880)/(1441+3880+6355+260)) # 44%