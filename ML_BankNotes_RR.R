
# ML PROJECT: ML Based Counterfeit Money Detaction. 
# Data used: Banknote_Autentication Dataset 
# Source: UCI ML Repository/ Kaggle dataset Repository/ ML Data Hub.
# Written By Mr. Murera Gisa.
#--------------------------------------------------------------

#Step 1: Set working Directory, Load the R Packages/ Libraries and the Dataset to be used
# Set Working Directory
setwd("C:/Users/Gisa/Desktop/ML PROJECT")

# Load R Packages ----------------------------------------

data_wrangling_packages <- c("tidyverse", "ggplot2", "xtable", "knitr","kableExtra", "openxlsx")
install.packages("caret", dependencies = c("Depends", "Suggests"))# Installing the caret that accommodate more than 200 ML algorithm.
install.packages("kernlab", dependencies = c("Depends", "Suggests")) # accomodate 10 MLA
machine_learning_packages <- c("MASS", "car", "kernlab","rpart","earth","gbm","xgboost","fastAdaboost","c50","randomForest","class","ada", "rda","e1071", "nnet","ipred", "dbarts", "klaR", "glmn","caretEnsemble")

#install.packages(c("kimr","caretEnsemble" ))

if(!require(install.load)){
  install.packages("install.load")
}

install.load::install_load(c(data_wrangling_packages, machine_learning_packages))
#load the package caret which accomodate more than 200 ML Algorithms+ others
install.packages("gridExtra")
install.packages("skimr")
install.packages("plotly")
install.packages("ggthemes") # package to change the entire appearence of plot 
library(ggthemes) # Load
library(tidyverse)
library(skimr)
library(plotly)
library(gridExtra)
library(install.load)
#---------------------------------------------------------------------

#Import the data set
BankNote <- read.csv("C:/Users/GISA/Desktop/ML PROJECT/BankNote.csv")

#------------
# Step 2: Data Preprocessing and Preparation + Structure of the dataframe

str(BankNote)
View(BankNote)
dim(BankNote)
summary(BankNote)
#------------------------
# Step 3: Data Visualization of importance of variables and Descriptive Statistics
# For Classification we need to convert the output into factors otherwise you are doing regression.

factor_variable_position <- c(5)
BankNote <- BankNote %>% mutate_at(.,vars(factor_variable_position),~as.factor(.))

View(BankNote)

BankNote1 <- BankNote %>% mutate(Y= as.factor(fct_recode(Y, Fake= "0",Genuine = "1")))
View(BankNote1)
#------------------
# The genuine and fake notes occurence percentage
#
Percentage <- round(prop.table(table(BankNote1$Y))*100, digits = 2) 
#Visualize dataframe
#------------------------
Graph<- BankNote1 %>% count(Y)%>% mutate(Rate=round(prop.table(table(BankNote1$Y))*100, digits = 2)) %>% ggplot(aes(x= Y, y= n, fill= Y))+ geom_bar(stat = "identity", width = 0.5, show.legend = FALSE)+ theme_bw()+ labs(x= "Bank Notes Status", y= "Bank Notes Volume", caption=  "Source: BankNotes_Authentication Data")+ scale_fill_manual(values = c("Fake Notes"= "Red", "Genuine Notes"=  "Green"), aesthetics = "fill")+ geom_text(aes(label=  str_c(Rate,"%")),vjust= 4.5,size= 2.5, color= "black")+ theme(legend.position= "top", axis.title.y= element_text(size= 12,face= "bold"),axis.title.x= element_text(size= 12,face= "bold"), axis.title.x= element_text(angle= 45,vjust= 0.3,face= "bold"))
library(plotly)
ggplotly(Graph, tooltip = c("x","y"))
#--------------------------ML MODEL FORMULATION---------

# Step 4: Splitting the data into Training and Test Sets

# Step 1: Get row numbers for the training data
partition <- createDataPartition(BankNote$Y, p = 0.8, list= FALSE)

# Step 2: Create the training  dataset
train_BankNote <- BankNote[partition,] #the training sample

cat("The dimension of the training set is (",dim(train_BankNote),")") # tuning the dimension of training sample

#Step 2: Create the test dataset
test_BankNote <- BankNote[-partition,] # the test sample

cat("The dimension of test set is (", dim(test_BankNote),")")

#---------------PREPROCESSING DATA------------

#Step 0: Some algorithms don't like missing values, so remove rows with missing values
#BankNote <- na.omit(BankNote)
# Step 1:  Rescale numeric features for being in similar range of values and it is applied to both training and test samples

preProcess_scale_model  <- preProcess(train_BankNote, method = c("center", "scale")) 

train_BankNote <- predict(preProcess_scale_model, train_BankNote) # rescaled train BankNote data (xytrain)
test_BankNote <- predict(preProcess_scale_model, test_BankNote)  #rescaled test BankNote data (xytest)

# Removing the output column to create the xtrain and xtest (If you have the categorical variables do the same to create dummies )

xtrain <- train_BankNote[-length(train_BankNote)] 
xtest <- test_BankNote[-length(test_BankNote)]

#-----------DESCRIPTIVE STATISTICS -------
library(skimr)

#kable() This serves to make a table in latex, html, markdown
#Ex: run the sumary of model by kable(summary(model)$coef, digits=2) or tab<-xtable(summary(model)$coef, digits=c(0, 2, 2, 1, 2))
#print(tab, type="html")

 
#skimmed <- skim_to_wide(xtrain) #Or xtrain %>% skim_to_wide()
#print(skimmed) # no NA values

DescriptiveStat <- skim(xtrain) #Or xtrain %>% skim()

Table <- kable(DescriptiveStat, type = "latex")
# Manipulation of skim()
xtrain_skimmed <- xtrain %>% skim_to_list()
xtrain_skimmed[["numeric"]] %>% dplyr::select( missing, mean, sd, hist)
xtrain_skimmed[["numeric"]] %>% dplyr::select( -p0, -p25, -p50, -p75, -p100)
# skim() with table by using kable()
library(pander)
library(xtable)
xtrain_skimmed[["numeric"]] %>% dplyr::select( missing, mean, sd, hist) %>% ktable()
xtrain_skimmed[["numeric"]] %>% dplyr::select( missing, mean, sd, hist) %>% xtable() # generating code for latex
xtrain_skimmed[["numeric"]] %>% dplyr::select( missing, mean, sd, hist) %>% pander()

#Step2: FEATURE SELECTION

#It serves to enumerate the features with low variance and zero variance.
#Eliminating low variance features

near_zero <- nearZeroVar(xtrain, freqCut = 95/5, uniqueCut = 10, saveMetrics = TRUE)

low_variance_cols <- near_zero[(near_zero$zeroVar == TRUE) | (near_zero$nzv == TRUE), ]

print(low_variance_cols) # This has shown that no feature has been qualified as low variance feature. then no removed teatures
xytrain <- train_BankNote
xytest <- test_BankNote

# Then store inputs X and outputs Y for later use.
xtrain <- xtrain 
ytrain <- xytrain$Y

xtest <- xtest
ytest <- xytest$Y

ntr <- nrow(xytrain)
nte <- nrow(xytest)

#-----EVALUATION METRICS :MCC METRICS

#---------METHOD 3--------Function with buggs fixed in large products

mcc <- function (actual, predicted)
{
  # Compute the Matthews correlation coefficient (MCC) score
  # Murera Gisa 15/05/2019
  # Geoffrey Anderson 10/14/2016 
  # Added zero denominator handling.
  # Avoided overflow error on large-ish products in denominator.
  #
  # actual = vector of true outcomes, 1 = Positive, 0 = Negative
  # predicted = vector of predicted outcomes, 1 = Positive, 0 = Negative
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



# -----------Step 5: Train and tuning the ML models

# See available ML algorithms in caret Ecosytem of models>200
modelnames <- paste(names(getModelInfo()), collapse = ',')
print(modelnames)

# To know the details of any model 
earth <- modelLookup('earth')
print(earth)

# Set the seed for reproducibility
set.seed(12345)
# Models to consider

# can also include  summaryFunction = multiClassSummary 
# possible validation methods are method= "cv", "boot632", "LGOCV","LOOCV","repeatedcv", "boot" 

#Resambling method
#control <- trainControl(method= "repeatedcv", number = 10, repeats= 3, classProbs= TRUE, summaryFunction = multiClassSummary)

# can be "Accuracy",   "logLoss", "ROC",   "Kappa"
#metric <- "ROC"

library(caret)
library(tidyverse)
library(InformationValue)
library(ROCR)
library(pROC)

# 1. ctree model (Conditional Inference Tree)
set.seed(12345)
model_ctree = train(Y ~., data = xytrain, method = 'ctree' ,trControl = trainControl(method = "repeatedcv", number = 10, repeats = 3, search = "grid"),  metric = "Accuracy")
# predict the outcome on a test set (Testing model or madel validation)
yhat_ctree <- predict(model_ctree, xytest)

#compare predicted outcome and true outcome by Confusion matrix
Conf_Mat <- table(yhat_ctree,ytest)
ClasRate <- sum(diag(Conf_Mat))/sum(Conf_Mat) #accuracy
# Model Performance Evaluation Metrics
#Computation of Mathews correratio coeficient
mcc_ctree <- mcc(ytest,yhat_ctree)
#AUROC_ctree <- AUROC(as.numeric(ytest), as.numeric(yhat_ctree))
#-----------------------------
#Testing the importance of variables in model

Importance <- varImp(model_ctree$finalModel)
chart <- plot(Importance, col ="blue", cex = 3, xlab = " Importance level of variables in model", ylab = " The data variables", main= " The ROC Importance of data variables", caption = "CtreeModel")
#---------------
# Computation of True classification rate
sum(diag(Conf_Mat)/sum(Conf_Mat))
# 2. lda model
set.seed(12345)
lda.model <- train(Y~., data = xytrain, method = "lda",trControl = trainControl(method = "repeatedcv", number = 10, repeats = 3, search = "grid"),  metric = "Accuracy") 
yhat.lda <- predict(lda.model, xytest)
Conf_Mat <- table(ytest, yhat.lda)
#Model Performance Evaluation
mcc_lda <- mcc(ytest, yhat.lda)
#AUROC_lda <-sensitivity(ytest, yhat.lda)
#--------------------------
resamples(list(LDA = lda.model, Ctree= model_ctree))
# 3. C5.0Tree (Single C5.0 Tree) model
set.seed(12345)
c5.model <- train(Y~., data = xytrain, method = "C5.0Tree",trControl = trainControl(method = "repeatedcv", number = 10, repeats = 3, search = "grid"),  metric = "Accuracy") 
yhat.c5 <- predict(c5.model, xytest)
Conf_Mat <- confusionMatrix(ytest, yhat.c5)
mcc_c5.0 <- mcc(ytest, yhat.c5)
#-----------------------------

# 4. naive_bayes model
set.seed(12345)
naiveb.model <- train(Y~., data = xytrain, method = "naive_bayes",trControl = trainControl(method = "repeatedcv", number = 10, repeats = 3, search = "grid"),  metric = "Accuracy") 
yhat.naiveb <- predict(naiveb.model, xytest)
Conf_Mat <- confusionMatrix(ytest, yhat.naiveb)
# Model Performance Evaluation
mcc_naive <- mcc(ytest, yhat.naiveb)

#AUROC(as.numeric(ytest),as.numeric(yhat.naiveb))
#---------------------------------

# 5. Kernel knn model
set.seed(12345)
kknn.model <- train(Y~., data = xytrain, method = "kknn",trControl = trainControl(method = "repeatedcv", number = 10, repeats = 3, search = "grid"),  metric = "Accuracy") 
yhat.kknn <- predict(kknn.model, xytest)
Conf_Mat <- confusionMatrix(ytest, yhat.kknn)

mcc_kknn <- mcc(ytest, yhat.kknn)
#AUROC(as.numeric(ytest),as.numeric(yhat.kknn))
#--------------------------

# 6. adaboost (Adaptive Boosting model) model
set.seed(12345)
adaboost_model <- train(Y~., data = xytrain, method = "adaboost", trControl = trainControl(method = "repeatedcv", number = 10, repeats = 3, search = "grid"),  metric = "Accuracy") 
yhat.adaboost <- predict(adaboost_model, xytest)
Conf_Mat <- confusionMatrix(ytest, yhat.adaboost)
mcc_adaboost <- mcc(ytest, yhat.adaboost)
#---------------Discussions-----
# Computation of True classification rate
sum(diag(Conf_Mat)/sum(Conf_Mat))




#-----------------------------------------------

# 7. cart (Classification and regression tree)
set.seed(12345)
caret_model <- train(Y~., data = xytrain, method= "rpart", trControl = trainControl(method = "repeatedcv", number = 10, repeats = 3, search = "grid"),  metric = "Accuracy") 

yhat.caret <- predict(caret_model, xytest)

Conf_mat <- confusionMatrix(ytest, yhat.caret)

mcc_cart <- mcc(ytest, yhat.caret)
#------------------------------------------

# 8.  Kernel Partial Least Square (PLS) model

pls.model <- train(Y~., data = xytrain, method = "kernelpls",trControl = trainControl(method = "repeatedcv", number = 10, repeats = 3, search = "grid"),  metric = "Accuracy") 
yhat.pls <- predict(pls.model, xytest)
Conf_Mat <- confusionMatrix(ytest, yhat.pls)
mcc_pls <- mcc(ytest, yhat.pls)
#--------------------------------------

#9. pda (Penanlized Discrimant Analysis) Model

pda.model <- train(Y~., data = xytrain, method = "pda",trControl = trainControl(method = "repeatedcv", number = 10, repeats = 3, search = "grid"),  metric = "Accuracy") 
yhat.pda <- predict(pda.model, xytest)
Conf_Mat <- confusionMatrix(ytest, yhat.pda)
mcc_pda <- mcc(ytest, yhat.pda)
#----------------------------------------

#10.  Rborist ( implements the Random Forest algorithm, with particular emphasis on high performance) Model

Rborist.model <- train(Y~., data = xytrain, method = "Rborist",trControl = trainControl(method = "repeatedcv", number = 10, repeats = 3, search = "grid"),  metric = "Accuracy") 
yhat.Rborist <- predict(Rborist.model, xytest)
Conf_Mat <- confusionMatrix(ytest, yhat.Rborist)
mcc_Rborist <- mcc(ytest, yhat.Rborist)
#-----------------------------------------------

# 11. xyf (Supervised Version Of Kohonen's Self-Organising Maps) model
set.seed(12345)
xyf.model <- train(Y~., data = xytrain, method = "xyf",trControl = trainControl(method = "repeatedcv", number = 10, repeats = 3, search = "grid"),  metric = "Accuracy") 
yhat.xyf <- predict(xyf.model, xtest)
Conf_Mat <- confusionMatrix(ytest, yhat.xyf)
mcc_xyf <- mcc(ytest, yhat.xyf)
#---------------------------

# 12. wsrf(Weighted Subspace Random Forest for Classification) model
set.seed(12345)
wsrf.model <- train(Y~., data = xytrain, method = "wsrf",trControl = trainControl(method = "repeatedcv", number = 10, repeats = 3, search = "grid"),  metric = "Accuracy") 
yhat.wsrf <- predict(wsrf.model, xtest)
Conf_Mat <- confusionMatrix(ytest, yhat.wsrf)
mcc_wsrf <- mcc(ytest, yhat.wsrf)

#-----------------

# 13. QdaCov (Robust Quadratic Discriminant Analysis) Model
set.seed(12345)
Qdacov.model <- train(Y~., data = xytrain, method = "QdaCov",trControl = trainControl(method = "repeatedcv", number = 10, repeats = 3, search = "grid"),  metric = "Accuracy") 
yhat.Qdacov <- predict(Qdacov.model, xytest)
Conf_Mat <- confusionMatrix(ytest, yhat.Qdacov)
mcc_Qdacov <- mcc(ytest, yhat.Qdacov)
#-------------------------

# 14. Pam ( Nearest Shrunken Centroids) model
pam.model <- train(Y~., data = xytrain, method = "pam",trControl = trainControl(method = "repeatedcv", number = 10, repeats = 3, search = "grid"),  metric = "Accuracy") 
yhat.pam <- predict(pam.model, xytest)
Conf_Mat <- confusionMatrix(ytest, yhat.pam)
mcc_pam <- mcc(ytest, yhat.pda)
#------------------------------------

# 15. Linda (Robust Linear Discriminant Analysis aka Constructor) model
set.seed(12345)
linda.model <- train(Y~., data = xytrain, method = "Linda",trControl = trainControl(method = "repeatedcv", number = 10, repeats = 3, search = "grid"),  metric = "Accuracy") 
yhat.linda <- predict(linda.model, xytest)
Conf_Mat <- confusionMatrix(ytest, yhat.linda)
mcc_linda <- mcc(ytest, yhat.linda)
#----------------------------------------

#16.  RFlda (High-Dimensional Factor-Based Linear Discriminant Analysis) Model
RFlda.model <- train(Y~., data = xytrain, method = "RFlda",trControl = trainControl(method = "repeatedcv", number = 10, repeats = 3, search = "grid"),  metric = "Accuracy") 
yhat.RFlda <- predict(RFlda.model, xytest)
Conf_Mat <- confusionMatrix(ytest, yhat.RFlda)
mcc_RFlda <- mcc(ytest, yhat.RFlda)
#------------------------------

# 17. slda (Stabilized Linear Discriminant Analysis) model
set.seed(12345)
slda.model <- train(Y~., data = xytrain, method = "slda",trControl = trainControl(method = "repeatedcv", number = 10, repeats = 3, search = "grid"),  metric = "Accuracy") 
yhat.slda <- predict(slda.model, xytest)
Conf_Mat <- confusionMatrix(ytest, yhat.slda)
mcc_slda <- mcc(ytest, yhat.slda)
#--------------------------

# 18. sda (Shrinkage Discriminant Analysis) model
set.seed(12345)
sda.model <- train(Y~., data = xytrain, method = "sda",trControl = trainControl(method = "repeatedcv", number = 10, repeats = 3, search = "grid"),  metric = "Accuracy") 
yhat.sda <- predict(sda.model, xytest)
Conf_Mat <- confusionMatrix(ytest, yhat.sda)
mcc_sda <- mcc(ytest, yhat.sda)
#---------------------------

#19. knn model
knn.model <- train(Y~., data = xytrain, method = "knn",trControl = trainControl(method = "repeatedcv", number = 10, repeats = 3, search = "grid"),  metric = "Accuracy") 
yhat.knn <- predict(knn.model, xytest)
Conf_Mat <- confusionMatrix(ytest, yhat.knn)
mcc_knn <- mcc(ytest, yhat.knn)
#--------------------------------------

#20. lvq (Learning Vector Quantization = special case of an artificial neural network) model
set.seed(12345)
lvq.model <- train(Y~., data = xytrain, method = "lvq",trControl = trainControl(method = "repeatedcv", number = 10, repeats = 3, search = "grid"),  metric = "Accuracy") 
yhat.lvq <- predict(lvq.model, xytest)
Conf_Mat <- confusionMatrix(ytest, yhat.lvq)
mcc_lvq <- mcc(ytest, yhat.lvq)
#------------------------------------------------

         #MODELS PERFORMANCE EVALUATION#


     #COMPARING MODELS BY mcc, accuracy, kohen,s kappa and logloss
     
#1.  mcc Evaluation metric to check the best model-------

evaluation <- tibble(Ctree = mcc_ctree, C5.0Tree = mcc_c5.0, Naive_Bayes = mcc_naive, KernelKnn = mcc_kknn, Adaboost = mcc_adaboost, CART = mcc_cart, Linda = mcc_linda,"QdaCov" = mcc_Qdacov, wsrf = mcc_wsrf, Lvq = mcc_lvq, SDA = mcc_sda, xrf = mcc_xyf, SLDA = mcc_slda )

mcc <- t(evaluation)

Model <- rownames(mcc)

comparision_table <- as_tibble(mcc) %>% add_column(Model) %>% rename(mcc = "V1") %>%  arrange(mcc) %>% add_column(SN = 1:length(Model),.after = 0) %>% dplyr::select(SN, Model, mcc)
#------------Writing data table-----------------
setwd("C:/Users/GISA/Desktop/ML PROJECT")
write.table(comparision_table, "ML_Performance.csv", sep=",", row.names = F)
# Visulaize the comparative table
Compadata <- read.csv("C:/Users/GISA/Desktop/ML PROJECT/ML_Performance.csv")

Plot<- ggplot(Compadata, aes(Model, mcc, fill= mcc)) + theme_bw()+ geom_bar(stat = "identity") + coord_polar() + labs(title = "MLA Mathews Correlation Coefficient", caption=  "Source: BankNotes_Authentication Data Set")+ theme(legend.position = "top", axis.title = element_blank()) +theme(legend.text = element_text(colour="blue", size=10,face= "bold")) + theme(legend.title = element_text(colour="red", size = 20,face="bold"))
#--------Save plot as png-

ggsave("comp_MLBanknote.png", width = 6.74, height = 4.54)
#----------------------------
# Model selection with maximum Mathews Correlation Coefficient.

maxmcc <- comparision_table %>% filter(mcc == max(mcc))
minmcc <- comparision_table %>% filter(mcc == min(mcc))

# Exporting the table in latex
Comparative_table_latex <- knitr::kable(comparision_table, "latex", caption = "Model Performance Evaluation by mcc Metric")

maxmcc_latex <- knitr::kable(maxmcc, "latex", caption = "Optimal Model")
minmcc_latex <- knitr::kable(minmcc, "latex", caption = "Worsen ML Model")

# Twiter HTML Table Attributes

kable(maxmcc, caption = 'Optimal model', align = c('c', 'c')) %>% kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive"))

#-------------
# 2. accuracy, kohen,s kappa and logloss

#calculate resamples from caret // exclude PLS

resample_results <- resamples(list(Ctree = model_ctree, C5.0Tree = c5.model, Naive_Bayes = naiveb.model, KernelKnn = kknn.model,Adaboost = adaboost_model, Cart = caret_model, Linda = linda.model,QdaCov = Qdacov.model, wsrf = wsrf.model, Lvq = lvq.model, slda = slda.model,SDA = sda.model, xrf = xyf.model))
# summarize the results
# print results to console

Results <- summary(resample_results,metric = c("Kappa","Accuracy"))
# plot Kappa values
densityplot(resample_results , metric = "Kappa" ,auto.key = list(columns = 3))
# plot all (higher is better)
bwplot(resample_results , metric = c("Kappa","Accuracy"))
# plot logLoss values boxplots (lower is better)
# similar to value or rank on the leaderboard
bwplot(resample_results , metric = "logLoss")
#-------------------------------------------------------------------------------------------------------------------------
#Predicting the class label by optimal adaboost mode 
xvalidation<- xytest
validation_label <- predict(adaboost_model, xvalidation)

# Convert to data frame

validation_label <- tibble(Y= validation_label)
validation_label<- validation_label %>% mutate(Y= as.factor(fct_recode(Y, Forged= "0",Genuine = "1")))

validation_table <- validation_label %>% count(Y) %>% mutate(Percent = round(n/sum(n)*100, digits=2))


#--------Class predictiol label-------- 
# Validation label prediction
xytest_class <- xytest %>% mutate(Y= as.factor(fct_recode(Y, Forged= "0",Genuine = "1")))

xytest_class <- xytest_class %>% mutate(Y= validation_label$Y) 
#-----Assess the agreement btn the actual and predicted

# Actual
ActualClass <- xytest_class %>% count(Y)%>% mutate(Percent=round(n/sum(n)*100, digits=2))
# Predicted
PredictedClass <- validation_label %>% count(Y) %>% mutate(Percent = round(n/sum(n)*100, digits=2))
#-------------------------------------------------------------------END--------------------------------------------------


library(ROCR) # library(pROC) STUDY AND CHECK
yhat.pda <- prediction(yhat.pda, ytest) #scores
summary(yhat.pda)

simple_ROC <- function(labels, scores) {
  labels<-labels[order(scores, decreasing = TRUE)]
  data.frame(TPR=cumsum(labels)/sum(labels), FPR= cumsum(!labels)/sum(!labels), labels)
}
#--------------------------
library(tidyverse)
prob.prediction <- function(yhat.model){
  as_tibble(yhat.model) %>% mutate(prob=if_else( genuine > forged, genuine, forged)) %>% add_column(ytest= as.numeric(if_else(ytest == "genuine", 0, 1))) 
}
#-------------------
library(caret)
lda.model <- train(Y~ ., data = xytrain, method = "lda")#, trControl = trainControl(method = "cv", number = 10, returnResamp = "all", classProbs = FALSE, summaryFunction = twoClassSummary), metric = "Accuracy")
library(ROCR)
install.packages("pROC")
library(pROC)
yhat.lda <- predict(lda.model, xytest, type = "prob")
head(yhat.lda)
roc(predictor =yhat.lda$1, response = ytest,levels= rev(levels(ytest)))



fg<- yhat.lda[xytrain$Y == 0]


bg <-yhat.lda[yhat.lda$Y== 1]
roc <- roc.curve(scores.class0 = fg, scores.class1 = bg, curve = T)
plot(roc)
min(as.vector(thisTrh), 1)
lda_data <- prediction(yhat.lda, ytest)
