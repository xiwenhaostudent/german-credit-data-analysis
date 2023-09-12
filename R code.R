library(ggplot2)
library(caret)
library(rpart)
library(rpart.plot)
library(tree)
library(e1071)
library(kernlab)      
library(neuralnet)
library(NeuralNetTools)
library(ROCR)
library(pROC)


G.data<-read.csv("/Users/GERMAN_DATA.csv",
                 header=TRUE)
str(G.data)
summary(G.data)

G.data$DUR<-as.numeric(G.data$DUR)
G.data$CRED_AMT<-as.numeric(G.data$CRED_AMT)
G.data$INST_RT_PER_DISP_INCM<-as.numeric(G.data$INST_RT_PER_DISP_INCM)
G.data$DUR_RES<-as.numeric(G.data$DUR_RES)
G.data$AGE<-as.numeric(G.data$AGE)
G.data$NUM_CRED<-as.numeric(G.data$NUM_CRED)
G.data$NUM_PEOP_LIABL<-as.numeric(G.data$NUM_PEOP_LIABL)
G.data$Y<-as.factor(G.data$Y)



sapply(G.data, function(x) sum(is.na(x)))



# split data into training and test sets
set.seed(800) 
index <- 1:nrow(G.data)
test_set_index <- sample(index, trunc(length(index)/3))
test_set <- G.data[test_set_index,]
train_set <- G.data[-test_set_index,]
train_set1 <- G.data[-test_set_index,]

# determine the max/min from the training set
d_max <- sapply(train_set[,c(2,5,8,11,13,16,18)], max)
d_min <- sapply(train_set[,c(2,5,8,11,13,16,18)], min)

# normalize the data to [0,1] use rescale function of scales package

# function for rescale the columns based on the training set max/min 
rescale <- function(dat, d_min, d_max) {
  c <- ncol(dat)
  for (i in 1:c) {
    dat[,i] <- sapply(dat[,i], function(x) (x - d_min[i])/(d_max[i] - d_min[i]))
  }
  return (dat)
}

# normalize the training/testing set
train_set[,c(2,5,8,11,13,16,18)] <- rescale(train_set[,c(2,5,8,11,13,16,18)], d_min, d_max)
test_set[,c(2,5,8,11,13,16,18)] <- rescale(test_set[,c(2,5,8,11,13,16,18)], d_min, d_max)




# Logistic Regression Model
log_fit1 <- train(Y~., data=train_set, method="glm", family="binomial")

summary(log_fit1)

# Prediction
pred<-predict(log_fit1, newdata=test_set)

confusionMatrix(pred,test_set$Y)



# 10-Fold Cross Validation:
ctrl <- trainControl(method = "repeatedcv", number = 10, savePredictions = TRUE)

log_fit2 <- train(Y~.,data=train_set, method="glm", family="binomial",
                  trControl = ctrl, tuneLength = 5)

summary(log_fit2)

# Prediction
pred <- predict(log_fit2, newdata=test_set)
confusionMatrix(data=pred, test_set$Y)
t1<-table(data=pred, test_set$Y)
log_acc<-(t1[1,1]+t1[2,2])/(t1[1,1]+t1[2,2]+t1[1,2]+t1[2,1])
log_spec<-t1[2,2]/(t1[1,2]+t1[2,2])



# Logistic ROC curve
log.predd <- predict(log_fit2, type='prob',test_set, probability = TRUE)

modelroc <- roc(test_set$Y,log.predd[,2])
plot(modelroc, type="S",print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     auc.polygon.col="skyblue", print.thres=TRUE, main="ROC Curve of Logistic Regression Model")





# kNN
ctrl <- trainControl(method = "repeatedcv", number = 10, savePredictions = TRUE)

knn_fit <- train(Y ~ ., data = train_set, method = "knn", 
                 trControl = ctrl, preProcess = c("center","scale"), tuneLength = 20)

# Output of kNN fit
knn_fit

# Plotting yields Number of Neighbours Vs accuracy (based on repeated cross validation)
plot(knn_fit)



knnPredict <- predict(knn_fit,newdata = test_set )
#Get the confusion matrix to see accuracy value and other parameter values
confusionMatrix(knnPredict, test_set$Y )
t1<-table(knnPredict, test_set$Y)
knn_acc<-(t1[1,1]+t1[2,2])/(t1[1,1]+t1[2,2]+t1[1,2]+t1[2,1])
knn_spec<-t1[2,2]/(t1[1,2]+t1[2,2])



# kNN ROC curve
knn.predd <- predict(knn_fit, type='prob',test_set, probability = TRUE)

modelroc <- roc(test_set$Y,knn.predd[,2])
plot(modelroc, type="S",print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     auc.polygon.col="skyblue", print.thres=TRUE, main="ROC Curve of kNN Model")




# Decision Tree 

trees <- tree(Y~., train_set)
plot(trees)
text(trees, pretty=0)

# Prediction and confusion matrix
treesPredict <- predict(trees,newdata = test_set , type="class")
confusionMatrix(treesPredict, test_set$Y )



# Cross validation and plot the tree
cv.trees <- cv.tree(trees, FUN = prune.misclass)
plot(cv.trees)




prune.trees <- prune.tree(trees, best=6)
plot(prune.trees)
text(prune.trees, pretty=0)



prune.treesPredict <- predict(prune.trees,newdata = test_set , type="class")
confusionMatrix(prune.treesPredict, test_set$Y )
t1<-table(prune.treesPredict, test_set$Y)
dt_acc<-(t1[1,1]+t1[2,2])/(t1[1,1]+t1[2,2]+t1[1,2]+t1[2,1])
dt_spec<-t1[2,2]/(t1[1,2]+t1[2,2])



# Decision Tree ROC curve
dt.predd <- predict(prune.trees,newdata = test_set)

modelroc <- roc(test_set$Y,dt.predd[,2])
plot(modelroc, type="S",print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     auc.polygon.col="skyblue", print.thres=TRUE, main="ROC Curve of Decsion Tree Model")




# Naive Bayes
ctrl <- trainControl(method = "repeatedcv", number = 10, savePredictions = TRUE)
nb_fit = train(train_set[,1:20],train_set[,21],'nb',
               trControl=ctrl)
nb_fit



nbPredict <- predict(nb_fit$finalModel,newdata = test_set[,1:20] )$class
#Get the confusion matrix to see accuracy value and other parameter values
confusionMatrix(nbPredict, test_set$Y )
t1<-table(nbPredict, test_set$Y)
nb_acc<-(t1[1,1]+t1[2,2])/(t1[1,1]+t1[2,2]+t1[1,2]+t1[2,1])
nb_spec<-t1[2,2]/(t1[1,2]+t1[2,2])



# Naive Bayes ROC curve
nb.predd <- predict(nb_fit$finalModel, type='prob',test_set, probability = TRUE)

modelroc <- roc(test_set$Y,nb.predd$posterior[,2])
plot(modelroc, type="S",print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     auc.polygon.col="skyblue", print.thres=TRUE, main="ROC Curve of Naive Bayes Model")




# Support Vector Machine

tc <- tune.control(cross = 10)
tune.out <- tune(svm, Y~.,
                 data = train_set, kernel = "radial",
                 ranges = list(cost = 10^(-1:2),
                               gamma = c(0.25,0.5,1,2,5)),
                 tunecontrol = tc)
summary(tune.out)
print(tune.out) # best parameters: cost=1, gamma=0.25





svm.prediction = predict(tune.out$best.model,newdata=test_set,type='class')
confusionMatrix(svm.prediction,as.factor(test_set$Y)) 
t1<-table(svm.prediction,as.factor(test_set$Y))
svm_acc<-(t1[1,1]+t1[2,2])/(t1[1,1]+t1[2,2]+t1[1,2]+t1[2,1])
svm_spec<-t1[2,2]/(t1[1,2]+t1[2,2])




# SVM ROC curve
svm_fit2 <- svm(Y~., data =train_set, cost=1, gamma=0.25, probability = TRUE)

svm.predd <- predict(svm_fit2, type='prob',test_set, probability = TRUE)

modelroc <- roc(test_set$Y,attr(svm.predd, "probabilities")[,2])
plot(modelroc, type="S",print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     auc.polygon.col="skyblue", print.thres=TRUE, main="ROC Curve of SVM Model")




# Neural Network
ctrl <- trainControl(method = "repeatedcv", number = 10, savePredictions = TRUE)

nn_fit <- train(Y~., data = train_set, 
                method = 'nnet', preProcess = c('center', 'scale'), trControl = ctrl,
                tuneGrid=expand.grid(size=c(10), decay=c(0.1)))





nnPredict <- predict(nn_fit,newdata = test_set )
#Get the confusion matrix to see accuracy value and other parameter values
confusionMatrix(nnPredict, test_set$Y )
t1<-table(nnPredict, test_set$Y)
nn_acc<-(t1[1,1]+t1[2,2])/(t1[1,1]+t1[2,2]+t1[1,2]+t1[2,1])
nn_spec<-t1[2,2]/(t1[1,2]+t1[2,2])



# Neural Network ROC curve
nn.predd <- predict(nn_fit, type='prob',test_set, probability = TRUE)

modelroc <- roc(test_set$Y,nn.predd[,2])
plot(modelroc, type="S",print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     auc.polygon.col="skyblue", print.thres=TRUE, main="ROC Curve of Neural Network Model")


# Comparison and Conclusions
cat("\n")
cat(" Logistic regression model's accuracy: ", log_acc,
    "; Specificity: ",log_spec ,"\n",
    "kNN model's accuracy: ", knn_acc,
    "; Specificity: ",knn_spec ,"\n",
    "Decision tree model's accuracy: ", dt_acc,
    "; Specificity: ",dt_spec ,"\n",
    "Naive Bayes model's accuracy: ", nb_acc,
    "; Specificity: ",nb_spec ,"\n",
    "Support vector machine model's accuracy: ", svm_acc,
    "; Specificity: ",svm_spec ,"\n",
    "Neural Network model's accuracy: ", nn_acc,
    "; Specificity: ",nn_spec ,"\n")




