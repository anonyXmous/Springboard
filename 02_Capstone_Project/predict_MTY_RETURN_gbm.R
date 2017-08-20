rm(list=ls())
gc(reset=TRUE)
setwd("D:/Rfiles/")
options(scipen=999) #prevent scientific notation
randomSeed = 123
set.seed(randomSeed)
library(gbm)
library(caret)

mape5 <- function(actual, preds) {
  mape = 0
  err = vector(mode="numeric", length=length(actual))
  for(i in 1:length(actual)) {
    err[i] = ifelse(actual[i] == 0, NA, (abs((actual[i] - preds[i])/actual[i])))
    mape = mape + err[i]
  }
  return (mape/length(err))
}

LogLossBinary = function(actual, predicted, eps = 1e-15) {  
  predicted = pmin(pmax(predicted, eps), 1-eps)  
  - (sum(actual * log(predicted) + (1 - actual) * log(1 - predicted))) / length(actual)
}

########################  MAIN ################
fname = 'c:/Users/bacoyjo/export.csv'

# target variable 
target_var = c('TRAVEL_TIME_HOURS')

# continous var
cont_var   = c('IS_40','IS_45','IS_GP','IS_HQ',  'VGM_WT','IS_OVERWT',  'IS_TOP_COMMODITY','BOX_COUNT','ETA_DD','ETA_MO')

# index var
index_var = c('CNTR_NUM', 'MTY_RTN_TRANSITTIME_ACTUAL')

ff_input <- read.csv(fname, stringsAsFactors = F)

ff_input$TRAVEL_TIME_HOURS = ff_input$MTY_RTN_TRANSITTIME_ACTUAL
ff_input_select = ff_input[,c(cont_var, target_var, index_var)]
for(i in c(cont_var, target_var))
  ff_input_select[,i]<-as.numeric(ff_input_select[,i])

str(ff_input_select)
#pairs(ff_input_select) 

#low_perct  = quantile(ff_input$TRAVEL_TIME_MINUTES, .25) - 1.5*IQR(ff_input$TRAVEL_TIME_MINUTES)
#high_perct = quantile(ff_input$TRAVEL_TIME_MINUTES, .75) + 1.5*IQR(ff_input$TRAVEL_TIME_MINUTES)
#ff_input = ff_input[ff_input$TRAVEL_TIME_MINUTES > low_perct & ff_input$TRAVEL_TIME_MINUTES< high_perct,]

# split data into training/testing set
trainIndex <- createDataPartition(ff_input_select$TRAVEL_TIME_HOURS, p = .8, list = F, times = 1)
training <- ff_input_select[trainIndex,]
training$TRAVEL_TIME_HOURS = training$MTY_RTN_TRANSITTIME_ACTUAL
testing <- ff_input_select[-trainIndex,]
testing$TRAVEL_TIME_HOURS = testing$MTY_RTN_TRANSITTIME_ACTUAL

#Ytrain = training[,target_var]
#Ytest  = testing[,target_var]
#training[,target_var] = NULL
#testing[,target_var]  = NULL
#testing_index = testing[,index_var]
#training[,index_var] =  NULL
#testing[,index_var]  =  NULL


##############GBM MODEL ####################
gbmModel = gbm(formula = TRAVEL_TIME_HOURS ~ IS_40 + IS_45 + IS_GP + IS_HQ + VGM_WT + IS_OVERWT + IS_TOP_COMMODITY + BOX_COUNT + ETA_DD + ETA_MO ,               distribution = "laplace",
               data = training,
               n.trees = 300,
               shrinkage = .05,
               cv.folds = 5,
               n.minobsinnode = 5)
gbmTrainPredictions = predict(object = gbmModel,
                              newdata = training,
                              n.trees = 300)

gbmTestPredictions = predict(object = gbmModel,
                             newdata = testing,
                             n.trees = 300)
head(data.frame("Actual" = testing$TRAVEL_TIME_HOURS,
                "PredictedProbability" = gbmTestPredictions))
summary(gbmModel, plot = FALSE)
bestTreeForPrediction = gbm.perf(gbmModel)
mape_err = mape5(as.vector(testing$TRAVEL_TIME_HOURS),as.vector(gbmTestPredictions))
sprintf("MAPE (percent):  %2.0f", (mape_err)*100)


################# L G O C V ###################
dataSubsetProportion = .2
randomRows = sample(1:nrow(training), floor(nrow(training) * dataSubsetProportion))
trainingHoldoutSet = training[randomRows, ]
trainingNonHoldoutSet = training[!(1:nrow(training) %in% randomRows), ]

trainingHoldoutSet$RowID = NULL
trainingNonHoldoutSet$RowID = NULL
trainingHoldoutSet$Model = NULL
trainingNonHoldoutSet$Model = NULL
gbmForTesting = gbm(formula = TRAVEL_TIME_HOURS ~ IS_40 + IS_45 + IS_GP + IS_HQ + VGM_WT + IS_OVERWT + IS_TOP_COMMODITY + BOX_COUNT + ETA_DD + ETA_MO ,               
                    distribution = "laplace",
                    data = trainingNonHoldoutSet,
                    n.trees = 500,
                    shrinkage = .01,
                    n.minobsinnode = 2)
summary(gbmForTesting, plot = FALSE)

gbmTestPredictions = predict(object = gbmForTesting,
                             newdata = testing,
                             n.trees = 500)
mape_err = mape5(as.vector(testing$TRAVEL_TIME_HOURS),as.vector(gbmTestPredictions))
sprintf("MAPE (percent):  %2.0f", (mape_err)*100)

gbmWithCrossValidation = gbm(formula = TRAVEL_TIME_HOURS ~ IS_40 + IS_45 + IS_GP + IS_HQ + VGM_WT + IS_OVERWT + IS_TOP_COMMODITY + BOX_COUNT + ETA_DD + ETA_MO ,               
                             distribution = "laplace",
                             data = trainingNonHoldoutSet,
                             n.trees = 500,
                             shrinkage = .1,
                             n.minobsinnode = 5, 
                             cv.folds = 5,
                             n.cores = 1)

bestTreeForPrediction = gbm.perf(gbmWithCrossValidation)

gbmTestPredictions = predict(object = gbmWithCrossValidation,
                             newdata = testing,
                             n.trees = 500)
mape_err = mape5(as.vector(testing$TRAVEL_TIME_HOURS),as.vector(gbmTestPredictions))
sprintf("MAPE (percent):  %2.0f", (mape_err)*100)
