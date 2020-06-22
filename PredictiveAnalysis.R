#####################################################################################
##                                                                                 ##
##                                 K-NN Analysis                                   ## 
##                                                                                 ##
#####################################################################################
# Install the required packages and load the libraries
install.packages("class")
install.packages("e1071")
install.packages("caret")
library(class)
library(e1071)
library(caret)


#####################################################################################
## 1. Import and partition the data into training (60%) and validation (40%) sets. ##
#####################################################################################
# Load the data
getwd()
setwd("/Users/mjchoi/Documents/DMBA-Assignment")
bank.df <- read.csv("UniversalBank.csv")

# Display the structure of the data frame
str(bank.df) 

# Summary overview of bank.df
summary(bank.df) 

# Explore the dataset 
sum(bank.df$Personal.Loan == 1) # 480 accepted the personal loan in the earlier campaign.
summary(bank.df[("Personal.Loan")]) # Refined summary overview of Personal.Loan, which is 9.6% 
t(t(names(bank.df)))

# Use set.seed() to ensure reproducibility of partitions
set.seed(111)  

# Generate a random sample of row numbers containing 60% of original data
train.index <- sample(row.names(bank.df), 0.6*dim(bank.df)[1])  
# Use setdiff() function to assign rows not in train.index to valid.index
valid.index <- setdiff(row.names(bank.df), train.index)  

# Use selected row numbers to populate training and validation data sets
# excluding ID and ZIP code
train.df <- bank.df[train.index, -c(1, 5)]  
valid.df <- bank.df[valid.index, -c(1, 5)]

# new customer
new.df <- data.frame(Age = 40, Experience = 10, Income = 84, Family = 2,
                     CCAvg = 2,Education = 2, Mortgage = 0, Securities.Account = 0,
                     CD.Account = 0, Online = 1, CreditCard = 1)

# Initialise normalised training, validation, complete data frames to originals 
train.norm.df <- train.df
valid.norm.df <- valid.df
bank.norm.df <- bank.df

# Use preProcess() function from caret package to normalise the data frame
norm.values <- preProcess(train.df[,-8], method=c("center", "scale"))

# Apply the model norm.values to scale and centre to the required columns of the 
# training, validation, the original data set and the new data we wish to classify.
# use predict() function to apply these transformations
train.norm.df[, -8] <- predict(norm.values, train.df[,-8])
valid.norm.df[, -8] <- predict(norm.values, valid.df[,-8])
bank.norm.df[,-c(1, 5, 10)] <- predict(norm.values, bank.df[,-c(1, 5, 10)])
new.norm.df <- predict(norm.values, new.df)


#####################################################################################
## 2.a. Perform a K-NN classification with all predictors except ID and ZIP code,  ##
##      using k = 1.                                                               ##
#####################################################################################
# Build the knn classification model using the given data 
knn.pred.1 <- class::knn(train.norm.df[,-8],
                         new.norm.df,
                         cl = train.norm.df$Personal.Loan, k = 1)


#####################################################################################
## 2.b. Specify the success class as 1 (loan acceptance), and use the default      ##
##      cut-off value of 0.5. How would a customer with the above characteristic   ##
##      be classified by this model?                                               ##
#####################################################################################
knn.pred.1

# Answer: 
# With 1 nearest neighbour, the new customer would be classified as 0; 
# he/she is not likely to accept a loan offer.


#####################################################################################
## 3. What is a choice of k that balances between overfitting and ignoring the     ##
##    predictor information?                                                       ##
#####################################################################################
# Initialise a data frame with two columns: k and accuracy.
accuracy.df <- data.frame(k = seq(1, 14, 1), accuracy = rep(0, 14))

# Compute knn for different k using a loop. 
for(i in 1:14){
  knn.pred <- knn(train.norm.df[, -8],
                  valid.norm.df[, -8],
                  cl = train.norm.df[, 8], k = i)
  accuracy.df[i, 2] <- confusionMatrix(knn.pred, as.factor(valid.norm.df[, 8]))$overall[1]
}
accuracy.df

# Answer:
# Output suggests that when k = 3, the highest level of accuracy is achieved 
# at 0.9575. Therefore, the best choice of k is 3.


#####################################################################################
## 4. Show the confusion matrix for the validation data that results from using    ##
##    the best value for k and interpret the results presented.                    ##
#####################################################################################
knn.pred.3 <- knn(train.norm.df[, -8],
                  valid.norm.df[, -8],
                  cl = train.norm.df[, 8], k = 3)

confusionMatrix(knn.pred.3, as.factor(valid.norm.df[, 8]))

# Answer:
# Accuracy of the confusion matrix is 0.9575. 
# This equals to the total number of True Positive (115) and True Negative (1800) 
# out of all (2000); (1800 + 115) / 2000
# True Positive Rate is 0.9934 (sensitivity).
# True Negative Rate is 0.6117 (specificity).


#####################################################################################
## 5. Using the customer profile given in task 2, classify the customer using the  ##
##    best value for k.                                                            ##
#####################################################################################
new.knn.pred <- class::knn(train.norm.df[,-8],
                           new.norm.df,
                           cl = train.norm.df[, 8], k = 3)
new.knn.pred

# Answer: The customer is classified as 0, therefore will not accept personal loan.


#####################################################################################
## 6. Repartition the data, this time into training, validation, and test sets     ##
##    (50%, 30% and 20%, respectively). Apply the k-NN method with the k chosen    ##
##    above. Compare the confusion matrix of the test set with that of the         ##
##    training and validation sets. Comment on the differences and their reason.   ##
#####################################################################################
set.seed(111)
# Generate a random sample of row numbers containing 50% of original data
train.index <- sample(row.names(bank.df), 0.5*dim(bank.df)[1])  
# Use setdiff() function to assign rows not in train.index to valid.index
valid.index <- sample(setdiff(row.names(bank.df), train.index), 0.3*dim(bank.df[1]))
# Assign the remaining 20% row serve as test
test.index <- setdiff(row.names(bank.df), union(train.index, valid.index))

# Create the 3 data frames by collecting all columns except ID and ZIP code
train.df <- bank.df[train.index, -c(1, 5)]  
valid.df <- bank.df[valid.index, -c(1, 5)]
test.df <- bank.df[test.index, -c(1, 5)]

# Initialise normalised training, validation, complete data frames to originals 
train.norm.df <- train.df
valid.norm.df <- valid.df
test.norm.df <- test.df

# Normalise data sets using preProcess() function 
norm.values <- preProcess(train.df[,-8], method=c("center", "scale"))

# apply the model norm.values to scale and centre to the required columns of the 
# training, validation, the original data set and the new data we wish to classify.
# use predict() function to apply these transformations
train.norm.df[, -8] <- predict(norm.values, train.df[, -8])
valid.norm.df[, -8] <- predict(norm.values, valid.df[, -8])
test.norm.df[, -8] <- predict(norm.values, test.df[, -8])

# k-NN of training set with k = 3
knn.pred.train <- class::knn(train.norm.df[,-8],
                             train.norm.df[,-8],
                             cl = train.norm.df[, 8], k = 3)

# Confusion matrix of training set
confusionMatrix(knn.pred.train, as.factor(train.norm.df[, 8])) 

# k-NN of validation set with k = 3
knn.pred.valid <- class::knn(train.norm.df[,-8],
                             valid.norm.df[,-8],
                             cl = train.norm.df[, 8], k = 3)

# Confusion matrix of training set
confusionMatrix(knn.pred.valid, as.factor(valid.norm.df[, 8])) 

# k-NN of test set with k = 3
knn.pred.test <- class::knn(train.norm.df[,-8],
                            test.norm.df[,-8],
                            cl = train.norm.df[, 8], k = 3)

# Confusion matrix of training set
confusionMatrix(knn.pred.test, as.factor(test.norm.df[, 8]))

# Answer:
# Train set accuracy: 0.9764
# Valid set accuracy: 0.9587
# Test set accuracy: 0.957
# Examining the confusion matrix of different data sets, the classifications is 
# most accurate on the training data set and least accurate on the test data set. 


####################################################################################
##                                                                                ##
##                            Naive Bayes Analysis                                ##
##                                                                                ##
####################################################################################
# Install the required packages and load the libraries
install.packages("reshape")
library(reshape)

bank.df <- read.csv("UniversalBank.csv")
#summary(bank.df)

# Convert to categorical variables for the naive bayes analysis
bank.df$Personal.Loan <- as.factor(bank.df$Personal.Loan)
bank.df$Online <- as.factor(bank.df$Online)
bank.df$CreditCard <- as.factor(bank.df$CreditCard)

set.seed(111)

# Create training and validation sets
selected.var <- c(13, 14, 10) # The variables chosen to perform analysis
# Generate a ranmdom sample of row numbers containing 60%  of original data
train.nb.index <- sample(c(1:dim(bank.df)[1]), dim(bank.df)[1]*0.6) 
train.nb.df <- bank.df[train.nb.index, selected.var]
valid.nb.df <- bank.df[-train.nb.index, selected.var]


#####################################################################################
## 7. Create a pivot table for the training data with Online as a column variable, ##
##    CC as a row variable, and Loan as a secondary row variable. The values       ##
##    inside the table should convery the count. Use functions melt() and cast(),  ##
##    or the function ftable().                                                    ##
#####################################################################################
# Use melt() to stack a set of columns into a single column of data
mlt <- melt(train.nb.df, id = c("CreditCard","Personal.Loan"), variable = c("Online"))
# Use cast() to reshape data; cast a molten data frame into a data frame 
casted <- cast(mlt, CreditCard + Personal.Loan ~ Online)
# Generate a pivot table
ftable(train.nb.df[,c(2, 3, 1)])


#####################################################################################
## 8. Looking at the pivot table, what is the probability that this customer will  ##
##    accept the loan offer? [This is the probability of loan accceptance          ##
##    (Loan = 1) conditional on having a bank credit card (CC = 1) and being an    ##
##    active user of online banking services (Online = 1)].                        ##
#####################################################################################

# Answer: 54/(54 + 497) = 9.8%


#####################################################################################
## 9. Create two separate pivot tables for the training data. One will have Loan   ##
##    (rows) as a function of Online (columns) and the other will have Loan (rows) ##
##    as a function of CC.                                                         ##
#####################################################################################
# Create a pivot table with Loan (row) and online (Column)
mlt.online <- melt(train.nb.df, id = c("Personal.Loan"), variable = "Online")
cast.online <- cast(mlt.online, Personal.Loan ~ Online)
ftable(train.nb.df[, c(3, 1)])

# Create a pivot table with Loan (row) and CC (Column)
mlt.cc <- melt(train.nb.df, id = c("Personal.Loan"), variable = "CreditCard")
cast.cc <- cast(mlt.cc, Personal.Loan ~ CreditCard)
ftable(train.nb.df[, c(3, 2)])


#####################################################################################
## 10. Calculate the following quantities [P(A|B)] means "the probability of A     ##
##     given B"]
#####################################################################################

# Answer:
# i. P(CC = 1 | Loan = 1)
# => 87/(87 + 205) = 29.8%

# ii. P(Online = 1 | Loan = 1)
# => 180/(112 + 180) = 61.6%

# iii. P(Loan = 1)
# => 292/3000 = 9.7%    # Either (205 + 87) or (112 + 180) equals 292

# iv. P(CC = 1 | Loan = 0)
# => 812/(1896 + 812) = 29.99% 

# v. P(Online = 1 | Loan = 0)
# => 1596/(1112 + 1596) = 58.9%

# vi.P(Loan = 0)
# => 2708/3000 = 90.3%    # either (1112 + 1596) or (1896 + 812) equals 2708


#####################################################################################
## 11. Use the quantities calculated above to calculate the Naive Bayes            ##
##     probability P(Loan = 1 | CC = 1, Online = 1).                               ##
#####################################################################################
prob <- ((87/(87 + 205))*(180/(112 + 180))*(292/3000)) /
  (((87/(87 + 205))*(180/(112 + 180))*(292/3000))
   + ((812/(1896 + 812))*(1596/(1112 + 1596))*(2708/3000)))
prob   # 0.1007717 

# Answer:
# Therefore, the Naive Bayes probability is 10.1%


#####################################################################################
## 12. Compare this value with the one obtained from the pivot table in task 8.    ##
##     Which is a more accurate estimate?                                          ##
#####################################################################################

# Probability estimate obtained from the Naive Bayes (10.1%) does not differ greatly
# from the exact method (9.8%). Naive Bayes method assumes no correlations 
# (independence) between columns, whereas the exact Bayes calulations relies on 
# finding records that share same predictor values as record-to-be-classified. 


#####################################################################################
## 13. Which of the entries in this table are needed for calculating               ##
##     P(Loan = 1 | CC = 1, Online = 1)? In R, run naive Bayes on the data.        ##
##     Examine the model output on training data and find the entry that           ##
##     corresponds to P(Loan = 1 | CC = 1, Online = 1).                            ##
##     Compare this to the number you obtained in task 11.                         ##
#####################################################################################
# Run naive bayes with Personal.Loan as the target variable and the columns 
# we will use as predictors (~ . means all columns will be used).
nb <- naiveBayes(Personal.Loan~., data = train.nb.df)
nb

nb.p <- (0.2979452 * 0.6164384 * 0.09733333)/(0.2979452 * 0.6164384 * 0.09733333 + 0.2998523 * 0.5893648)
nb.p # 9.2%

# The Naive Bayes does not differ greatly from the result obtained in task 11. 
