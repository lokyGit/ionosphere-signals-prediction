library(caret)
library(ggplot2)
library(rpart)
library(rpart.plot)
library(rattle)
library(partykit)
library(pROC)
library(Boruta)
library(nnet)
library(ROCR)
library(class)

#dataset=read.csv("~/Library/Mobile Documents/com~apple~CloudDocs/NEU/Classes/Summer 2020/Predictive Analytics/Final Project/ionosphere.data", header=FALSE)
dataset=read.csv(file.choose(), header = F)
dataset$V35=as.factor(dataset$V35)
str(dataset)
dataset$V1 = as.numeric(as.character(dataset$V1))
str(dataset)
dim(dataset)

# applying boruta method to identify the variables importance
Boruta_model = Boruta(V35 ~ ., data = dataset, doTrace=2)
Boruta_model
Boruta_model$finalDecision


#removing the unimportant variables from train set

dataset.sig = dataset[-c(2)]
dim(dataset.sig)
str(dataset.sig)
set.seed(1234)
ind= sample(2, nrow(dataset.sig), replace=TRUE, prob=c(0.7, 0.3))
dataset.train = dataset.sig[ind==1,]
dataset.test = dataset.sig[ind==2,]
View(dataset.sig)

#KNN Model (Tian’s part)
#Creating seperate dataframe for 'Creditability' feature which is our target.
train.label = dataset.train[,34]
test.label =dataset.test[,34]
#build the marchine learning model
sqrt(nrow(dataset.train))
#the square of 245 is 15.6,so we create 2 model
knn.15 = knn(train=dataset.train[,-34], test=dataset.test[,-c(34:35)], cl=train.label, k=15)
knn.16 = knn(train=dataset.train[,-34], test=dataset.test[,-c(34:35)], cl=train.label, k=16)

#model evaluation
ACC.15 = 100 * sum(test.label == knn.15)/NROW(test.label)
ACC.16 = 100 * sum(test.label == knn.16)/NROW(test.label)
ACC.15  # 77.55102
ACC.16  # 76.53061
#As shown above, the accuracy for K = 15 is 77.55 and for K = 16it is the same. 
#We can also check the predicted outcome against the actual value in tabular form:
table(knn.15,test.label)

#Now, let’s use the confusion matrix to calculate the accuracy of the KNN model with K value set to 15:
confusionMatrix(knn.15 ,test.label) #accuracy: 77.55%

#So, from the output, we can see that our model predicts the outcome with an accuracy of 78.57% which 
# A point to remember is that the more data (optimal data) 
#you feed the machine, the more efficient the model will be.
knn.test=dataset.test 
knn.test$pred= cbind(knn.15)
library(ROCR)
#ROC
knn.pred = prediction(knn.test$pred, test.label)
knn.roc = performance(knn.pred,"tpr","fpr")
plot(knn.roc, main = "ROC Curve for KNN")
abline(a=0, b=1)

#AUC
knn.auc = performance(knn.pred,"auc")
knn.auc=unlist(slot(knn.auc,"y.values"))
knn.auc=round(knn.auc,4)
legend(0.6,0.3, knn.auc, title ='AUC')



#GLM Model - Lokesh Palacharla
#fitting logistic regression to the ionosphere dataset.

glm_model=glm(V35~. , family=binomial(link = "logit"), 
              data = dataset.train)
summary(glm_model)

# predicting on test set

test_pred = predict(glm_model, dataset.test, type = "response")
test_prediction =  ifelse(test_pred>0.5 , "g", "b")
test_pred_tab = table(Predicted = test_prediction, Actual = dataset.test$V35)
test_prediction_error =  1 - sum(diag(test_pred_tab))/sum(test_pred_tab)
error_percentage_test = round(test_prediction_error*100,2)
prediction_accuracy_test = round((1 - test_prediction_error)*100,2) #  Accuracy: 92.86%

# Confusion Matrix for test set

test_factor_pred=as.factor(test_prediction)
glm.cm_test = confusionMatrix(test_factor_pred,dataset.test$V35)

# ROC for train set

pred.test =  ifelse(test_prediction == "g", 1, 0)
ionosphere.pred = prediction(pred.test, dataset.test$V35)
ionosphere.roc = performance(ionosphere.pred,"tpr","fpr")
plot(ionosphere.roc, main = "ROC Curve for GLM")
abline(a=0, b=1)

#AUC

ionosphere.auc = performance(ionosphere.pred,"auc")
ionosphere.auc=unlist(slot(ionosphere.auc,"y.values"))
ionosphere.auc=round(ionosphere.auc,4)
legend(0.6,0.3, ionosphere.auc, title ='AUC')



#Decision Tree
dataset.rpart = rpart(V35~ .,
                       data = dataset.train, 
                       method='class',
                       control = rpart.control(minsplit = 10))
print(dataset.rpart)
printcp(dataset.rpart)
plotcp(dataset.rpart)
summary(dataset.rpart)
dataset.rpart$variable.importance
fancyRpartPlot(dataset.rpart,digits = 3, caption=NULL,cex=0.49)

test=predict(dataset.rpart, dataset.test[,-34], type='class')
rpart.test=dataset.test
mean(rpart.test$V35==test)
rpart.test$test= cbind(test)
table(predict=test,actual=rpart.test$V35)

confusionMatrix(rpart.test$V35,test) # ACCURACY: 80.61%


rpart.test$test =as.numeric(rpart.test$test)
rpart.test$V35=as.numeric(rpart.test$V35)
str(rpart.test)
plot(roc(rpart.test$V35,rpart.test$test), print.auc=F, auc.polygon=TRUE, grid=c(0.1, 0.2),
     max.auc.polygon=TRUE,
     print.thres=F,
     main="ROC Curve for Rpart")
rpart.auc = performance(prediction(rpart.test$test,rpart.test$V35),"auc")
rpart.auc=unlist(slot(rpart.auc,"y.values"))
rpart.auc=round(rpart.auc,4)
text(0.25, 0.2, labels=sprintf("AUC: %0.4f", labels=rpart.auc), col="black")

MAE= function(actual, predicted){mean(abs(actual-predicted))}
MAE
MAE(rpart.test$V35,rpart.test$test)  #0.1938776


# Implimenting ensamble methods and creating the submodels
# Loading the required libraries
library(mlbench)
library(caret)
library(caretEnsemble)

control = trainControl(method="repeatedcv", number=20, repeats=10, savePredictions=TRUE, classProbs=TRUE)
algorithmList = c('rpart', 'glm', 'knn')
set.seed(1234)
models = caretList(V35~., data=dataset.sig, trControl=control, methodList=algorithmList)
results = resamples(models)
summary(results)
dotplot(results)

# correlation between results
modelCor(results)
splom(results)
# creating a stack using glm
stackControl = trainControl(method="repeatedcv", number=10, repeats=3, savePredictions=TRUE, classProbs=TRUE)
set.seed(1234)
stack.glm = caretStack(models, method="glm", metric="Accuracy", trControl=stackControl)
print(stack.glm)  #Accuracy - 90.97816%

# stack using rpart
set.seed(1234)
stack.rpart = caretStack(models, method="rpart", metric="Accuracy", trControl=stackControl)
print(stack.rpart) #Accuracy - 92.54511%


# stack using random forest
set.seed(1234)
stack.rf = caretStack(models, method="rf", metric="Accuracy", trControl=stackControl)
print(stack.rf) #Accuracy: 95.20418%


# stack using knn
set.seed(1234)
stack.knn = caretStack(models, method="knn", metric="Accuracy", trControl=stackControl)
print(stack.knn) # Accuracy: 94.94777%



