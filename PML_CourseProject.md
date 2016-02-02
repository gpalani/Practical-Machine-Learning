# Pratical Machine Learning Course Project
gpalani  
January 31, 2016  

## Background
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).


## Question
Predict the manner in which the participants exercise. 

The question being asked defines our model and how we can set up the model and predict it, then test it and see if we can validate it accurately. The template ofcourse has been laid out i.e. ask the questions, get the input data, define the features that can predict , create the model(algorithm) and evalute it.

Dumbbell Biceps Curl in five different fashions: 
1. exactly according to the specification (Class A)
2. throwing the elbows to the front (Class B) 
3. lifting the dumbbell only halfway (Class C)
4. lowering the dumbbell only halfway (Class D) 
5. throwing the hips to the front (Class E)


### Loading the data from the provided URL 


```r
library(caret)
traindataURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testdataURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
training <- read.csv(url(traindataURL), na.strings=c("NA","#DIV/0!",""))
testing <- read.csv(url(testdataURL), na.strings=c("NA","#DIV/0!",""))
```

Doing some preprocessing to increase the data quality. Removing zero covariates.


```r
#Cleaning the variable
NAindex <- apply(training,2,function(x) {sum(is.na(x))}) 
training <- training[,which(NAindex == 0)]
NAindex <- apply(testing,2,function(x) {sum(is.na(x))}) 
testing <- testing[,which(NAindex == 0)]

#Preprocessing Variables

i <- which(lapply(training, class) %in% "numeric")
preObj <-preProcess(training[,i],method=c('knnImpute', 'center', 'scale'))
trainingSet <- predict(preObj, training[,i])
trainingSet$classe <- training$classe
testSet <-predict(preObj,testing[,i])
```
Removing near Zero varible that increases the prediction ability. We do this on both the training and test data partitions.


```r
# Remove nonZero variables
nzv_training <- nearZeroVar(trainingSet, saveMetrics=TRUE)
trainingSet <- trainingSet[,nzv_training$nzv==FALSE]
nzv_testing <- nearZeroVar(testSet, saveMetrics=TRUE)
testSet  <- testSet[,nzv_testing$nzv==FALSE]
```

We use the createDataPartiton to divide the training and test data. I use 75% split for the training data/test data split.


```r
set.seed(343435)
inTrain <- createDataPartition(trainingSet$classe, p =.75, list = FALSE)
# Splitting the data
exer_training_data <- trainingSet[inTrain,]
exer_testing_data <- testSet[-inTrain,]
```

## Model

The model choosen is the random forest method for getting the best possible predictability. 


```r
modelFit <- train(classe ~. ,data = exer_training_data,method = "rf", trControl=trainControl(method='cv'), number=5, allowParallel=TRUE )
modelFit
```

```
## Random Forest 
## 
## 14718 samples
##    27 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 13246, 13246, 13248, 13245, 13246, 13245, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
##    2    0.9934772  0.9917482  0.002606925  0.003298930
##   14    0.9929332  0.9910601  0.002797636  0.003540343
##   27    0.9906914  0.9882242  0.003299097  0.004174723
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 2.
```

Check the accuracy of the training data 


```r
trainingPred <- predict(modelFit, exer_training_data)
confusionMatrix(trainingPred,exer_training_data$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4185    0    0    0    0
##          B    0 2848    0    0    0
##          C    0    0 2567    0    0
##          D    0    0    0 2412    0
##          E    0    0    0    0 2706
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9997, 1)
##     No Information Rate : 0.2843     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1839
## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1839
## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1839
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```

Now run the prediction on the partition test data. 


```r
#testPred<- predict(modelFit, exer_testing_data)
#confusionMatrix(testPred,exer_testing_data$classe)
```

## RESULTS

Run the prediction on the actual test data 


```r
classeTest<- predict(modelFit, testSet)
classeTest
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
