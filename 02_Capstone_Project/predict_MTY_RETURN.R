rm(list=ls())
gc(reset=TRUE)

setwd("D:/Rfiles/")
options(scipen=999) #prevent scientific notation

library(h2o)
library(leaps)
library(kernlab)   
library(caret)
library(readr)
library(dplyr)
library(lubridate)
library(Hmisc)
library(ggplot2)
library(Metrics)
library(DMwR)
library(methods)
library(reshape2)
library(dummies)
library(xgboost)
library(e1071)
library(doParallel) #Register workers for parallel run

mape5 <- function(actual, preds) {
  err = vector(mode="numeric", length=length(actual))
  for(i in 1:length(actual)) {
    err[i] = ifelse(actual[i] == 0, NA, (abs((actual[i] - preds[i])/actual[i]))*100.0)
  }
  return (err)
}

# GridSearch regression
GridSearch_regression <- function(model.label,
                                  Xtrain, 
                                  Ytrain,
                                  Xtest,
                                  GridObject,
                                  ControlObject,
                                  Importance = FALSE, 
                                  Verbose = FALSE)
{
  if (model.label == "knn") {  ### KNN_Reg
    if(!missing(GridObject)) {
      knnGrid = GridObject
    } else {
      knnGrid = expand.grid(c(.k = 1:20))       
    } 
    model <- caret::train(x = Xtrain, y = Ytrain,  
                          method    = "knn", 
                          preProc   = c("center", "scale"), # or preProcess = 'range'
                          tuneGrid  = knnGrid,
                          trControl = controlObject,
                          importance= Importance,
                          verbose   = Verbose)
  } else if (model.label == "svmRadial") {  ### KNN_Reg
    if(!missing(GridObject)) {
      svmGrid = GridObject
    } else {
      svmGrid =expand.grid(.C = c(1, 10, 100, 500, 1000),
                           .sigma = c(0.001, 0.01, 0.1))      
    } 
    model <- caret::train(x = Xtrain, y = Ytrain,  method = "svmRadial", 
                          preProc   = c("center", "scale"), # or preProcess = 'range'
                          tuneGrid  = svmGrid,
                          trControl = controlObject,
                          importance= Importance,
                          verbose   = Verbose)
  } else if(model.label == "randomForest") {    ### RANDOM FOREST ####
    if(!missing(GridObject)) {
      rfGrid = GridObject
    } else {
      rfGrid = expand.grid(mtry = c(1,2,3,5,7,9))
    }
    model <- caret::train(x = Xtrain, y = Ytrain, 
                          method     = "rf",
                          tuneGrid   = rfGrid,
                          ntrees     = 200,
                          #max_depth = 7, min_child_weight = 5, 
                          do.trace   = 100,
                          trControl  = controlObject,
                          importance = Importance,
                          verbose    = Verbose)
    
  } else if (model.label == "xgbLinear") {
    print("Start xgbLinear")
    if(!missing(GridObject)) {
      xgbGrid = GridObject 
    } else {
      xgbGrid = expand.grid(nrounds= c(5, 10, 100),
                            eta    = c(0.01, 0.05, 0.1, 0.3, 0.5), # step size shrinkage 
                            lambda = c(0), # L2 Regularization 
                            alpha  = c(1))  # L1 Regularization 
    } 
    model <- caret::train(x = as.matrix(Xtrain), y = as.numeric(Ytrain), method = "xgbLinear",
                          tuneGrid  = xgbGrid,
                          nthread   = 6,
                          eval_metric = "rmse",
                          subsample = 0.8,
                          gamma = .5,
                          preProcess = c("center","scale"), # scale feature
                          #preProcess="pca",  # another scale feature
                          trControl = controlObject,
                          importance= Importance,
                          verbose   = Verbose)
  } else if(model.label == "svmPoly") {
    print("Start svmPoly")
    if(!missing(GridObject)) {
      xgbGrid = GridObject 
    } else {
      svmPolyGrid =  expand.grid(C      = c(10, 100, 200),
                                 scale  = c(0.01),
                                 degree = c(2,3,4))
    }
    model <- caret::train(x = Xtrain, y = Ytrain,  method = "svmPoly", 
                          verbose = T, 
                          preProc   = c("center", "scale"), # or preProcess = 'range'
                          tuneGrid  = svmPolyGrid,
                          trControl = controlObject,
                          importance= Importance,
                          verbose   = Verbose)
  } else if(model.label == "pcr") {
    print("Start pcr")
    if(!missing(GridObject)) {
      pcrGrid   = GridObject
    } else {
      pcrGrid = expand.grid(.ncomp = 1:10)     
    }
    pcrGrid
    model <- caret::train(x = Xtrain, y = Ytrain,  method = "pcr", 
                          verbose = T, 
                          preProc   = c("center", "scale"), # or preProcess = 'range'
                          tuneGrid  = pcrGrid,
                          trControl = controlObject,
                          importance= Importance, 
                          verbose   = Verbose)
  }
  plot(model)
  model$results  # results of training
  model$bestTune # tuning parameters
  return(model)
}

###################################### main ######################################
fname = 'c:/Users/bacoyjo/export.csv'

# target variable 
target_var = c('TRAVEL_TIME_HOURS')

# continous var
cont_var   = c('IS_40','IS_45','IS_GP','IS_HQ',  'VGM_WT','IS_OVERWT',  'IS_TOP_COMMODITY','BOX_COUNT','ETA_DD','ETA_MO')

# index var
index_var = c('CNTR_NUM', 'MTY_RTN_TRANSITTIME_ACTUAL')

ff_input <- read.csv(fname, stringsAsFactors = F)

dim(ff_input)
names(ff_input)
describe(ff_input)
str(ff_input)

# Generate features
ff_input$TRAVEL_TIME_HOURS = ff_input$MTY_RTN_TRANSITTIME_ACTUAL

ff_input_select = ff_input[,c(cont_var, target_var, index_var)]
for(i in c(cont_var, target_var))
  ff_input_select[,i]<-as.numeric(ff_input_select[,i])

str(ff_input_select)

pairs(ff_input_select) 

# split data into training/testing set
set.seed(123)
trainIndex <- createDataPartition(ff_input_select$TRAVEL_TIME_HOURS, p = 0.8, list = F, times = 1)
training <- ff_input_select[trainIndex,]
testing <- ff_input_select[-trainIndex,]

Ytrain = training[,target_var]
Ytest  = testing[,target_var]
curr_tt = ff_input[-trainIndex,]$MTY_RTN_TRANSITTIME_ACTUAL
training[,target_var] = NULL
testing[,target_var]  = NULL
testing_index = testing[,index_var]
training[,index_var] =  NULL
testing[,index_var]  =  NULL
names(training)
names(testing)

cat('***************************** models **************************************\n')
cl<-makeCluster(6)
registerDoParallel(cl)

controlObject <- trainControl(method = "cv", number = 10, returnResamp = "all", search = "grid", verboseIter = TRUE, allowParallel = TRUE)
xgbGrid = expand.grid(nrounds= c(5, 10, 100), 
                      eta    = c(0.01, 0.05, 0.1, 0.3, 0.5), # learning rate
                      lambda = c(0), # L2 Regularization 
                      alpha  = c(1)) # L1 Regularization

rfGrid  = expand.grid(mtry = ceil(c(0.1,0.25,0.5)*length(names(training))))

model='xgbLinear'
Grid = xgbGrid
Importance = TRUE 
model_fit  = GridSearch_regression(model,
                                   Xtrain = training,
                                   Ytrain = Ytrain,
                                   Xtest  = testing,
                                   GridObject = Grid,
                                   ControlObject = controlObject,
                                   Importance = Importance,
                                   Verbose = FALSE)

plot(varImp(model_fit))
if(Importance) {
  imports <- varImp(model_fit)$importance %>% 
    mutate(names=row.names(.)) %>%
    arrange(-Overall)
}
ggplot(model_fit) + theme(legend.position = "top")
pred = predict(model_fit , newdata = testing)
err  = c(regr.eval(Ytest, pred, stats=c('mape','mae','rmse')), 'R2'=caret::R2(Ytest, pred, form='traditional'))
mape_err = mape5(as.vector(Ytest),as.vector(pred))
ma_err   = abs(pred-Ytest)
eval   = cbind(testing_index, ACTUAL_TT=(Ytest),PRED_TT=pred,  MAPE=mape_err, MA = ma_err) 
eval
result = list("err" = err, "imports" = imports, "eval" = eval)
result
err

####################################
model='randomForest'
Grid = rfGrid
model_fit_rf  = GridSearch_regression(model,
                                      Xtrain = training,
                                      Ytrain = Ytrain,
                                      Xtest  = testing,
                                      GridObject = Grid,
                                      ControlObject = controlObject,
                                      Importance = Importance,
                                      Verbose = FALSE)

plot(varImp(model_fit_rf))
if(Importance) {
  imports <- varImp(model_fit_rf)$importance %>% 
    mutate(names=row.names(.)) %>%
    arrange(-Overall)
}
ggplot(model_fit_rf) + theme(legend.position = "top")
pred = predict(model_fit_rf , newdata = testing)
err  = c(regr.eval((Ytest),  (pred), stats=c('mape','mae','rmse')), 'R2'=caret::R2(Ytest, pred, form='traditional'))
mape_err = mape5(as.vector(Ytest),as.vector(pred))
ma_err   = abs((pred)-(Ytest))

eval   = cbind(testing_index, ACTUAL_TT=(Ytest),PRED_TT=pred,  MAPE=mape_err, MA = ma_err) 
result = list("err" = err, "imports" = imports, "eval" = eval)
eval   = cbind(testing_index, ACTUAL_TT=(Ytest),PRED_TT=pred,  MAPE=mape_err, MA = ma_err, CURR_TT=curr_tt) 

eval
err
stopCluster(cl)
