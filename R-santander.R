library("ggplot2")
library("Scale")
library("psych")
library("gplots")
library("pROC")
library("ROSE")
library("corrgram")
library(e1071)
library("C50")
library("xgboost")


rm(list=ls())
setwd("D:/Edwisor/Project/Santander project")
getwd()

train=read.csv("train.csv", header = T)
test=read.csv("test.csv",header=T)

#Data understanding
dim(train)
str(train)
train[1:5,1:5]
unique(train$target)
table(train$target)

dim(test)
str(test)
test[1:5,1:5]

#Missing value Analysis
sum(is.na(train))
sum(is.na(test))

#Outlier Analysis
train$target= as.factor(train$target)
Numeric = sapply(train, is.numeric)
Numeric_var= train[, Numeric]
cnames= colnames(Numeric_var)
#for (i in 1:length(cnames))
#   {assign(paste0("gn",i), ggplot(aes_string(y = (cnames[i])), data = subset(train))+ 
#              stat_boxplot(geom = "errorbar", width = 0.5))}
#Removing outliers
#for(i in cnames){
#   print(i)
#   val = train[,i][train[,i] %in% boxplot.stats(train[,i])$out]
#   #print(length(val))
#   train = train[which(!train[,i] %in% val),]}

#Dimesionality Reduction 

#Principal componenet analysis

df=train
rownames(df)=df$ID_code
df= subset(df, select = -c(ID_code,target))
##Standardisation
cnames=colnames(df)
for(i in cnames){
  df[,i] = (df[,i] - mean(df[,i]))/sd(df[,i])}
principal_comp= prcomp(df, scale. = T)
summary(principal_comp)
pr_var =principal_comp$sdev
#computing variance
pr_var=pr_var^2
#Proportion of variance explained
prop_var= pr_var/sum(pr_var)
#Plots
par(mfrow=c(1,2))
plot(prop_var, xlab="principal component", ylab="prop. of variance explained")
plot(cumsum(prop_var),xlab="pricipal component", ylab = "cumulative proportion of variance")
#Cumulative proportion plot comes out as straight line that means our data is already gone through principal component analysis

#Correlation analysis
train_correlations=cor(train[,cnames])
min(train_correlations)
max(train_correlations)
#No independent variable have high correlation with other independent variable

#_____________________________________#Feature Scaling#________________________________________#

# Data is normally distributed hence we can do standarization for feature scaling
for(i in cnames){
  train[,i]= (train[,i] - mean(train[,i]))/sd(train[,i])}

for(i in cnames){
  test[,i]= (test[,i] - mean(test[,i]))/sd(test[,i])}

rownames(train)=train$ID_code
rownames(test)=test$ID_code
train=subset(train, select = -ID_code)
test=subset(test, select = -ID_code)

#__________________________#Spliting Data into test and validation#__________________________#

train_index=sample(1:nrow(train),0.8*nrow(train))
x_train=train[train_index,]    
x_test=train[-train_index,]

#________________________#Model Prepration and Hyper parameter tunning#______________________#
getmodel_accuracy=function(conf_matrix){
  print(conf_matrix)
  model_parm =list()
  tn =conf_matrix[1,1]
  tp =conf_matrix[2,2]
  fp =conf_matrix[1,2]
  fn =conf_matrix[2,1]
  p =(tp)/(tp+fp)
  r =(tp)/(tp+fn)
  fpr= (fp)/(fp+tp)
  fnr=(fn)/(fn+tn)
  print(paste("accuracy",round((tp+tn)/(tp+tn+fp+fn),2)))
  print(paste("precision",round(p ,2)))
  print(paste("recall",round(r,2)))
  print(paste("fpr",round(fpr,2)))
  print(paste("fnr",round(fnr,2)))
}

#_____________Logistic Regression_______________#
logit_model= glm(target~. , data = x_train, family = "binomial")
logit_pred = predict(logit_model, newdata= x_test, type= "response")
logit_prediction= ifelse(logit_pred >0.5,1,0)
conf_matrix=table(x_test[,1],logit_prediction)
getmodel_accuracy(conf_matrix)
roc=roc(x_test[,1],logit_prediction)
auc(roc)
plot(roc ,main ="Logistic Regression Roc ")
#Results
#logit_pred:"accuracy 0.92","precision 0.69","recall 0.28","fpr 0.01","fnr 0.72","AUC 0.6321"
#    0     1
#0 35534   489
#1  2872  1105

#________________Logistic regression with random over sampling_________________#

x_train_over= ovun.sample(target~.,data=x_train, method="over")$data
table(x_train_over$target)

over_logit =glm(formula = target~. ,data =x_train_over ,family='binomial')
summary(over_logit)
LB_prob_over =predict(over_logit , newdata=x_test ,type = 'response' )
LB_pred_over = ifelse(LB_prob_over >0.5, 1, 0)
conf_matrix= table(x_test[,1] , LB_pred_over)
getmodel_accuracy(conf_matrix)
roc=roc(x_test[,1],LB_pred_over)
auc(roc)
plot(roc ,main ="LR based Roc with over sampling")
#Results
#LB_pred_over: "accuracy 0.78", "precision 0.28", "recall 0.78","fpr 0.22","fnr 0.22",AUC 0.7798
#  0     1
#0 28169  7854
#1   884  3093 

#_____________Naive Baye____________#
x_train$target= as.factor(x_train $target)
x_test$target = as.factor(x_test$target)
NB_model= naiveBayes(target~., data= x_train)
NB_pred= predict(NB_model, x_test[,-1],type="class")
conf_matrix=table(x_test[,1],NB_pred)
getmodel_accuracy(conf_matrix)
x_train$target= as.numeric(x_train $target)
NB_pred=as.numeric(NB_pred)
roc=roc(x_test[,1],NB_pred)
auc(roc)
plot(roc ,main ="Naive baye Roc ")
#Results
#NB_pred:"accuracy 0.92",precision 0.71","recall 0.37", "fpr 0.02","fnr 0.63",AUC 0.6745
#   0     1
#0 35426   597
#1  2523  1454

#________________Naive Baye with random over sampling_________________#
x_train_over$target= as.factor(x_train_over$target)
NB_over=naiveBayes(target~., data=x_train_over)
NB_pred_over=predict(NB_over, x_test[,-1] ,type='class')
conf_matrix=table(x_test[,1],NB_pred_over)
getmodel_accuracy(conf_matrix)
x_train$target= as.numeric(x_train $target)
NB_pred_over=as.numeric(NB_pred_over)
roc=roc(x_test[,1],NB_pred_over)
auc(roc)
plot(roc ,main ="Naive baye Roc with oversampling ")
#Results
#NB_pred_over:"accuracy 0.81","precision 0.32","recall 0.8","fpr 0.19","fnr 0.2",AUC 0.8054
#      0     1
# 0 29259  6764
# 1   801  3176

#____________Decision Tree______________#
x_train$target= as.factor(x_train $target)
DT_model=C5.0(target~.,x_train)
DT_pred=predict(DT_model,x_test[,-1],type="class")
conf_matrix=table(x_test[,1],DT_pred)
x_train$target= as.numeric(x_train $target)
DT_pred=as.numeric(DT_pred)
getmodel_accuracy(conf_matrix)
roc=roc(x_test[,1],DT_pred)
auc(roc)
plot(roc ,main ="DT Roc ")
#Resuts
#DT_pred:"accuracy 0.88","precision 0.26", "recall 0.12","fpr 0.74","fnr 0.09","AUC 0.5426"
#    0     1
#0 34603  1406
#1  3495   496

#________________Decision Tree with random over sampling_________________#
DT_model=C5.0(target~.,x_train_over)
DT_pred=predict(DT_model,x_test[,-1],type="class")
conf_matrix= table(x_test[,1] , DT_pred)
getmodel_accuracy(conf_matrix)
DT_pred=as.numeric(DT_pred)
roc=roc(x_test[,1], DT_pred)
auc(roc)
plot(roc ,main="DT ROC with Over sampling")
#Results
#DT_model with over sampled data:"accuracy 0.83","precision 0.17","recall 0.19","fpr 0.83","fnr 0.09","AUC:0.5451"
#    0     1
#0 32238  3771
#1  3213   778


#___________XGBoost_____________#
x_train$target= as.numeric(as.factor(x_train$target))-1 
x_test$target=as.numeric(as.factor(x_test$target))-1 
x_train_over$target =as.numeric(as.factor(x_train_over$target))-1 

# coverting data into dmatrix (required in xgboost) 
x_train_DM =xgb.DMatrix(data =as.matrix(x_train[,-1]),label= x_train$target)
x_test_DM =xgb.DMatrix(data=as.matrix(x_test[,-1]) ,label  =x_test$target)
x_train_over_DM =xgb.DMatrix(data =as.matrix(x_train_over[,-1]) ,label=x_train_over$target)

xgb1 = xgb.train(data = x_train_DM,nrounds = 500,eta = 0.1,max.depth = 3,scale_pos_weight =2,
                 objective = "binary:logistic")
XG_prob =predict(xgb1 , as.matrix(x_test[,-1] ) )
XG_pred = ifelse(XG_prob >0.5, 1, 0)
conf_matrix= table(x_test[,1] ,XG_pred)
getmodel_accuracy(conf_matrix)
roc=roc(x_test[,1], XG_pred )
auc(roc)
plot(roc ,main="xgboost ROC")
#Results
# XG_pred:"accuracy 0.92","precision 0.7","recall 0.34","fpr 0.3","fnr 0.07","AUC 0.6621"
#     0     1
#0 35427   582
#1  2633  1358

#___________XGBoost Using oversampled data________________#
xgb1_over = xgb.train(data = x_train_over_DM,nrounds = 500,eta = 0.1,max.depth = 3,scale_pos_weight=2,
                 objective = "binary:logistic")
XG_prob_over =predict(xgb1_over , as.matrix(x_test[,-1] ))
XG_pred_over = ifelse(XG_prob_over >0.5, 1, 0)
conf_matrix= table(x_test[,1] , XG_pred_over)
getmodel_accuracy(conf_matrix)
roc=roc(x_test[,1],XG_pred_over)
auc(roc)
plot(roc ,main="xgboost ROC with oversampled data")
#Results
# XG_pred_over:"accuracy 0.71","precision 0.24","recall 0.88","fpr 0.76","fnr 0.02","AUC 0.7867"
#     0     1
#0 24906 11103
#1   472  3519


#_____________________Model Selection_______________________#

#XG_pred_over:"accuracy 0.71","precision 0.24","recall 0.88","fpr 0.76","fnr 0.02","AUC 0.7867"
#XG_pred:"accuracy 0.92","precision 0.7","recall 0.34","fpr 0.3","fnr 0.07","AUC 0.6621"
#DT_pred_over:"accuracy 0.83","precision 0.17","recall 0.19","fpr 0.83","fnr 0.09","AUC:0.5451"
#DT_pred:"accuracy 0.88","precision 0.26", "recall 0.12","fpr 0.74","fnr 0.09","AUC 0.5426"
#NB_pred_over:"accuracy 0.81","precision 0.32","recall 0.8","fpr 0.19","fnr 0.2",AUC 0.8054
#NB_pred:"accuracy 0.92",precision 0.71","recall 0.37", "fpr 0.02","fnr 0.63",AUC 0.6745
#LB_pred_over: "accuracy 0.78", "precision 0.28", "recall 0.78","fpr 0.22","fnr 0.22",AUC 0.7798
#logit_pred:"accuracy 0.92","precision 0.69","recall 0.28","fpr 0.01","fnr 0.72",AUC 0.6321

#The model we should select should have high ROC_AUC, Recall and precision , apart from that
#model must have low False negative and false positive rate
#The model we develop using Naive baye with over sampled data is good regarding above statement compaire
# to other model

#_______________predicting new test data__________________#

test$target=predict(NB_over,test,type='class')
Result_final=subset(test, select =target)


#_________________wrighting Output______________#
write.csv(Result_final,"test_predicted_R.csv",row.names = T)
