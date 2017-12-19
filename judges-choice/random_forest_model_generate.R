# Clean environment and load required packages
rm(list=ls())
cat("\014")
gc()

#Read training file
setwd("~/drivendata/fish/")
file_name = "crop_data_v3_32x32.csv"

library(data.table)
train_dataset <- fread(file_name)

# Set width
w <- 32
# Set height
h <- 32

library(caret)

train_dataset$label <- factor(train_dataset$label)

# Set names
names(train_dataset) <- c("label",paste("pixel",c(1:(w*h)),sep=""))

#Data Partition
set.seed(8)
intrain = createDataPartition(train_dataset$label, p = 0.9,list=FALSE)   #partition 90% of training data for training
training = train_dataset[intrain,]
testing = train_dataset[-intrain,]

library(randomForest)
set.seed(100)
model_rf <- randomForest(label~.,data=training)
pred <- predict(model_rf,testing)
confusion_matrix <- table(pred,testing$label)
confusionMatrix(confusion_matrix)