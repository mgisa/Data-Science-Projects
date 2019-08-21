# ML PROJECT: Bank of Kigali Credit Scoring . 
# Data used: BK Loan Dataset 
# Written By Mr. Murera Gisa.
# _________________________________

#Step 1: Set working Directory, Load the R Packages/ Libraries and the Dataset to be used
# Set Working Directory
setwd("C:/Users/Gisa/Desktop/ML PROJECT")

#Loading libraries
library(caret)
library(ggthemes) 
library(tidyverse)
library(skimr)
library(plotly)
library(gridExtra)
library(doSNOW)
library(doParallel)
library(MLmetrics)
library(parallel)
library(iterators)
library(doParallel)
library(foreach)
library(doSNOW)
library(install.load)

#Importing data
Loandata <- read.csv("C:/Users/Gisa/Desktop/ML PROJECT/LOANDATA_BK.csv", stringsAsFactors = FALSE, header = TRUE)
View(Loandata)
sum(is.na(Loandata))
# Function for detecting NA observations: 
na_rate <- function(x) {x %>% is.na() %>% sum() / length(x)}

sapply(Loandata, na_rate) %>% round(2)

#Function to compute mode in data
Mode <- function (x, na.rm) {
  xtab <- table(x)
  xmode <- names(which(xtab == max(xtab)))
  if (length(xmode) > 1) xmode <- ">1 mode"
  return(xmode)
}
#Loop to impute the missing data by mode

for (var in 1:ncol(Loandata)) {
  if (class(Loandata[,var])=="numeric") {
    Loandata[is.na(Loandata[,var]),var] <- mean(Loandata[,var], na.rm = TRUE)
  } else if (class(Loandata[,var]) %in% c("character", "factor")) {
    Loandata[is.na(Loandata[,var]),var] <- Mode(Loandata[,var], na.rm = TRUE)
  }
}
#Viewing data to check whether model class were imputed 
View(Loandata)

#Visualization

counting_creditscore <- Loandata %>% count(Credit.Score.Group) %>% mutate(Rate=round(prop.table(table(Loandata$Credit.Score.Group))*100, digits = 2)) %>% ggplot(aes(x=Credit.Score.Group , y= n, fill = Credit.Score.Group)) + geom_bar(stat = "identity", width = 0.5, show.legend = FALSE) + theme_bw() + labs(x= "Credit score categories", y= "Customers' total number", caption =  "Source: Bk_Credit_Scoring") + scale_fill_manual(values = c("Excellent" = "green", "Good"=  "blue", "Poor"= "red"), aesthetics = "fill")+scale_x_discrete(limits = c("Poor","Excellent","Good")) + ggtitle("Credit Scoring") + geom_text(aes(label=  str_c(Rate,"%")),vjust= 4.5,size= 2.5, color= "black")+theme(legend.position= "top", axis.text.x = element_text(angle = 45, face = "bold", colour = "black", size = 15),axis.title.x= element_text(size = 12,face = "bold"), axis.title.y = element_text(angle = 90,vjust = 0.3,face = "bold")) 

plotly_build(counting_creditscore)
ggsave("counting_creditscore.png", width = 6.74, height = 4.54)
#______________________________________
Province <- Loandata %>% count(Province) %>% mutate(Rate=round(prop.table(table(Loandata$Province))*100, digits = 2)) %>% ggplot(aes(x=Province , y= n, fill = Province)) + geom_bar(stat = "identity", width = 0.5, show.legend = FALSE) + theme_bw() + labs(x= "Province", y= "Customers' total number", caption =  "Source: Bk_Credit_Scoring") + scale_fill_manual(values = c("Kigali City" = "green", "West"=  "blue", "East"= "red", "South"="cyan", "North" = "yellow", "Diaspora"="magenta"), aesthetics = "fill")+scale_x_discrete(limits = c("Diaspora","North","South", "West","East", "Kigali City")) + ggtitle("Credit distribution in provinces") + geom_text(aes(label=  str_c(Rate,"%")),vjust= 4.5,size= 2.5, color= "black")+theme(legend.position= "top", axis.text.x = element_text(angle = 45, face = "bold", colour = "black", size = 15),axis.title.x= element_text(size = 12,face = "bold"), axis.title.y = element_text(angle = 90,vjust = 0.3,face = "bold")) 
plotly_build(Province)
ggsave("Province.png", width = 6.74, height = 4.54)
#______________________________________
Returning <- Loandata %>% count(ReturningCustomer) %>% mutate(Rate=round(prop.table(table(Loandata$ReturningCustomer))*100, digits = 2)) %>% ggplot(aes(x=ReturningCustomer , y= n, fill = ReturningCustomer)) + geom_bar(stat = "identity", width = 0.5, show.legend = FALSE) + theme_bw() + labs(x= "Returning customer groups", y= "Customers' total number", caption =  "Source: Bk_Credit_Scoring") + scale_fill_manual(values = c("NO" = "red", "YES"="green"), aesthetics = "fill")+ scale_x_discrete(limits = c("YES","NO")) + ggtitle("Number of customers who has returned to request a credit ") + geom_text(aes(label=  str_c(Rate,"%")),vjust= 4.5,size= 2.5, color= "black")+theme(legend.position= "top", axis.text.x = element_text(angle = 45, face = "bold", colour = "black", size = 15),axis.title.x= element_text(size = 12,face = "bold"), axis.title.y = element_text(angle = 90,vjust = 0.3,face = "bold")) 
plotly_build(Returning)
ggsave("Returning.png", width = 6.74, height = 4.54)
#______________________________________
#ML MODELS 

#summarize the credit score distribution
percentage <- prop.table(table(Loandata$Credit.Score.Group)) * 100
cbind(freq=table(Loandata$Credit.Score.Group), percentage=percentage)

Loandata %>% mutate(ReturningCustomer= factor(Loandata$ReturningCustomer, levels=c("YES","NO"),labels= c("1","0")))

Loandata <- Loandata %>% mutate(ReturningCustomer= as.factor(fct_recode(ReturningCustomer, NO= "0",YES = "1")))

#Removing the unmeaningful columns on MLA
Loandata<-Loandata[,-(1:3)]
Loandata1<-Loandata[,-12]
dim(Loandata1)
factor_variable_position <- c(12)
Loandata1 <- Loandata1 %>% mutate_at(.,vars(factor_variable_position),~as.factor(.))
#MODEL BUILDING

#1. Split the data set
# Create the training and test datasets for Loandata

# Step 1: Get row numbers for the training data
partition <- createDataPartition(Loandata1$Credit.Score.Group, p = 0.8, list = FALSE)

# Step 2: # Create the training sample
train_Loandata <- Loandata1[partition, ]

cat("The dimension of the training set is (", dim(train_Loandata), ")")
# Step 3: # Create the test sample
test_Loandata <- Loandata1[-partition, ]

cat("The dimension of test set is (", dim(test_Loandata), ")")

# Scaling the continuous variables

preProcess_scale_model <- preProcess(train_Loandata, method = c("center", "scale"))

# Here is what preProcess_scale_model does
# It only normalized the 11 continuous variables.
print(preProcess_scale_model)

train_Loandata <- predict(preProcess_scale_model, train_Loandata)

test_Loandata <- predict(preProcess_scale_model, test_Loandata)

# Removing the class column on train data to be able to select the variable

xtrain <- train_Loandata[-length(train_Loandata)]
xtest <- test_Loandata[-length(test_Loandata)]


#FEATURE SELECTION
# Eliminate low variance features

near_zero <- nearZeroVar(xtrain, freqCut = 95 / 5, uniqueCut = 10, saveMetrics = TRUE)

low_variance_cols <- near_zero[(near_zero$zeroVar == TRUE) | (near_zero$nzv == TRUE), ]

print(low_variance_cols)
# Remove low variance columns on train set

xtrain <- xtrain %>% dplyr::select(-Paid.Penalty)
# Appending Y to the  dataset

xytrain <- bind_cols(xtrain, y = train_Loandata$Credit.Score.Group)

# Remove low variance columns on test set

xtest <- xtest %>% dplyr::select(-Paid.Penalty)

# Appending Y to the dataset

xytest <- bind_cols(xtest, y = test_Loandata$Credit.Score.Group)
#___________________________________________________
#STORE X and Y 
# Store X and Y for later use.

xtrain <- xtrain
ytrain <- xytrain$y

xtest <- xtest
ytest <- xytest$y

ntr <- nrow(xytrain)
nte <- nrow(xytest)
#_________________________________________
#EVALUATION METRICS :MCC METRICS

#---------METHOD 3--------Function with buggs fixed in large products

mcc <- function (actual, predicted)
{
  # function returns MCC
  
  TP <- sum(actual == 1 & predicted == 1)
  TN <- sum(actual == 0 & predicted == 0)
  FP <- sum(actual == 0 & predicted == 1)
  FN <- sum(actual == 1 & predicted == 0)
  #TP;TN;FP;FN # for debugging
  sum1 <- TP+FP
  sum2 <- TP+FN
  sum3 <- TN+FP
  sum4 <- TN+FN
  
  denom <- as.double(sum1)*sum2*sum3*sum4 # as.double to avoid overflow error on large products
  if (any(sum1==0, sum2==0, sum3==0, sum4==0)) {
    denom <- 1
  }
  mcc <- ((TP*TN)-(FP*FN)) / sqrt(denom)
  return(mcc)
}
#_________________________________________

# ML Models to predict Probability of default

# Set conditions for training model and cross-validation: 

set.seed(123)
number <- 5
repeats <- 5
control <- trainControl(method = "repeatedcv", 
  number = number, 
  repeats = repeats, 
  classProbs = TRUE, 
  savePredictions = "final", 
  index = createResample(xytrain$y, number*repeats), 
  summaryFunction = multiClassSummary, 
  allowParallel = TRUE)
#________________________
#TRAINING MODELS

# 1.  C5.0Tree (Single C5.0 Tree) model
model_c5.0 <- train(y~., data = xytrain, method = 'C5.0Tree' ,trControl = control, metric = "Accuracy")
# predict the outcome on a test set (Testing model or model validation)
yhat_c5.0 <- predict(model_c5.0, xytest)
#Computation of Mathews correration coeficient
mcc_c5.0 <- mcc(ytest,yhat_c5.0)
#___________________________

# 2. sda model
model_sda <- train(y~., data = xytrain, method = 'sda' ,trControl = control, metric = "Accuracy")
yhat_sda <- predict(model_sda, xytest)
mcc_sda <- mcc(ytest,yhat_sda)
#___________________________

# 3. lvq model
model_lvq <- train(y~., data = xytrain, method = 'lvq' ,trControl = control, metric = "Accuracy")
yhat.lvq <- predict(model_lvq, xytest)
mcc_lvq <- mcc(ytest,yhat_lvq)
#___________________________

# 4. slda model
model_slda <- train(y~., data = xytrain, method = 'slda' ,trControl = control, metric = "Accuracy")
yhat_slda <- predict(model_slda, xytest)
mcc_slda <- mcc(ytest,yhat_slda)
#___________________________

# 5. QdaCov (Robust Quadratic Discriminant Analysis) model
model_QdaCov <- train(y~., data = xytrain, method = 'QdaCov' ,trControl = control, metric = "Accuracy")
yhat.Qdacov <- predict(model_QdaCov, xytest)
mcc_Qdacov <- mcc(ytest,yhat.Qdacov)
#___________________________

# 6. CART (Classification and Regression Tree) model
model_cart <- train(y~., data = xytrain, method = 'rpart' ,trControl = control, metric = "Accuracy")
yhat.cart <- predict(model_cart, xytest)
mcc_cart <- mcc(ytest,yhat.cart)
#___________________________

# 7. Naive Bayes model
naiveb.model <- train(y~., data = xytrain, method = 'naive_bayes' ,trControl = control, metric = "Accuracy")
yhat.naiveb <- predict(naiveb.model, xytest)
mcc_naiveb <- mcc(ytest,yhat.naiveb)
#___________________________

# 8. Linda(Robust Linear Discriminant Analysis aka Constructor) model
Linda.model <- train(y~., data = xytrain, method = 'Linda' ,trControl = control, metric = "Accuracy")
yhat.linda <- predict(Linda.model, xytest)
mcc_linda<- mcc(ytest,yhat.linda)
#___________________________

#9. xyf(Supervised Version Of Kohonen's Self-Organising Maps) model
xyf.model <- train(y~., data = xytrain, method = 'xyf' ,trControl = control, metric = "Accuracy")
yhat.xyf <- predict(xyf.model, xytest)
mcc_xyf <- mcc(ytest,yhat.xyf)
#___________________________

#10. Ctree (Conditional Inference Tree) Model 
ctree.model <- train(y~., data = xytrain, method = 'ctree' ,trControl = control, metric = "Accuracy")
yhat_ctree <- predict(ctree.model, xytest)
mcc_ctree <- mcc(ytest,yhat_ctree)
#___________________________

#11. WSRF (Weighted Subspace Random Forest for Classification) model 
wsrf.model <- train(y~., data = xytrain, method = 'wsrf' ,trControl = control, metric = "Accuracy")
yhat.wsrf <- predict(wsrf.model, xytest)
mcc_wsrf <- mcc(ytest,yhat.wsrf)
#___________________________

#12. Kernel knn  model 
kknn.model <- train(y~., data = xytrain, method = 'wsrf' ,trControl = control, metric = "Accuracy")
yhat.kknn <- predict(kknn.model, xytest)
mcc_kknn <- mcc(ytest,yhat.kknn)
#___________________________

#13. Adaboost  model 
adaboost.model <- train(y~., data = xytrain, method = 'wsrf' ,trControl = control, metric = "Accuracy")
yhat.adaboost <- predict(adaboost.model, xytest)
mcc_adaboost <- mcc(ytest,yhat.adaboost)
#___________________________


#COMPARING MODELS BY mcc

# mcc Evaluation metric to check the optimal model

evaluation <- tibble(Ctree = mcc_ctree, C5.0Tree = mcc_c5.0, Naive_Bayes = mcc_naiveb, KernelKnn = mcc_kknn, Adaboost = mcc_adaboost, CART = mcc_cart, Linda = mcc_linda,"QdaCov" = mcc_Qdacov, wsrf = mcc_wsrf, Lvq = mcc_lvq, SDA = mcc_sda, xrf = mcc_xyf, SLDA = mcc_slda )

mcc <- t(evaluation)

Model <- rownames(mcc)

comparision_table <- as_tibble(mcc) %>% add_column(Model) %>% rename(mcc = "V1") %>%  arrange(mcc) %>% add_column(SN = 1:length(Model),.after = 0) %>% dplyr::select(SN, Model, mcc)

#------------Writing data table in working directory-------------

write.table(comparision_table, "ML_Performance.csv", sep=",", row.names = F)

# Visualize the comparative table
Compadata <- read.csv("C:/Users/GISA/Desktop/ML PROJECT/ML_Performance.csv")

Plot<- ggplot(Compadata, aes(Model, mcc, fill= mcc)) + theme_bw()+ geom_bar(stat = "identity") + coord_polar() + labs(title = "MLA Mathews Correlation Coefficient", caption=  "Source: BankNotes_Authentication Data Set")+ theme(legend.position = "top", axis.title = element_blank()) +theme(legend.text = element_text(colour="blue", size=10,face= "bold")) + theme(legend.title = element_text(colour="red", size = 20,face="bold"))

#--------Save plot in png format

ggsave("comp_MLBanknote.png", width = 6.74, height = 4.54)
#----------------------------
# Model selection with maximum Mathews Correlation Coefficient.

maxmcc <- comparision_table %>% filter(mcc == max(mcc))

# Exporting the table in latex
Comparative_table_latex <- knitr::kable(comparision_table, "latex", caption = "Model Performance Evaluation by mcc Metric")

maxmcc_latex <- knitr::kable(maxmcc, "latex", caption = "Optimal Model")
minmcc_latex <- knitr::kable(minmcc, "latex", caption = "Worsen ML Model")
#_______________________END OF R SCRIPT_________________
