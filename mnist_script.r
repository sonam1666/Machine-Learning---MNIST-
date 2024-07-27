# TODO: Insert the path to the directory that holds the MNIST data files
setwd("C:/Users/chava/Documents/Machine Learning/MNIST-data/")
library(caret)
library(class)
library(factoextra)
library(tidyverse)
library(GGally)
library(caTools)
library(rpart)
library(rpart.plot)
library(randomForest)
load_mnist <- function() {
  load_image_file <- function(filename) {
    ret = list()
    f = file(filename,'rb')
    readBin(f,'integer',n=1,size=4,endian='big')
    ret$n = readBin(f,'integer',n=1,size=4,endian='big')
    nrow = readBin(f,'integer',n=1,size=4,endian='big')
    ncol = readBin(f,'integer',n=1,size=4,endian='big')
    16
    x = readBin(f,'integer',n=ret$n*nrow*ncol,size=1,signed=F)
    ret$x = matrix(x, ncol=nrow*ncol, byrow=T)
    close(f)
    ret
  }
  load_label_file <- function(filename) {
    f = file(filename,'rb')
    readBin(f,'integer',n=1,size=4,endian='big')
    n = readBin(f,'integer',n=1,size=4,endian='big')
    y = readBin(f,'integer',n=n,size=1,signed=F)
    close(f)
    y
  }
  train_dataset <<- load_image_file('train-images-idx3-ubyte')
  test_dataset <<- load_image_file('t10k-images-idx3-ubyte')
  train_dataset$y <<- load_label_file('train-labels-idx1-ubyte')
  test_dataset$y <<- load_label_file('t10k-labels-idx1-ubyte')
  
  list(train = train_dataset, test = test_dataset)
}
show_digit <- function(arr784, col=gray(12:1/12), ...) {
  image(matrix(arr784, nrow=28)[,28:1], col=col, ...)
}
17
# load data
mnist_data <- load_mnist()
train_dataset <- mnist_data$train
test_dataset <- mnist_data$test
str(mnist_data)
# inspect contents
summary(train_dataset$x)
summary(train_dataset$y)
# how the pictures looks like
train_dataset$x[1,]
show_digit(train_dataset$x[1,])
# having a look at the individual features (pixels)
pairs(test_dataset$x[,404:408],
      col=c("red","green","blue","aquamarine","burlywood","darkmagenta","chartreuse","yellow
","chocolate","darkolivegreen")
      [test_dataset$y+1])
# note for visualization: add one to labels (0 labels in the original table make R skip the
entry, which creates an inconsistent coloring)
# some pixels correlate very strongly, other don't
18
C <- cov(train_dataset$x)
image(C, main = "Pixel Correlations")
dim(train_dataset$x)
dim(test_dataset$x)
######################################################################
##########
# Normalize the data
x_train <- train_dataset$x / 255
y_train <- train_dataset$y
x_test <- test_dataset$x / 255
y_test <- test_dataset$y
#Check for missing values
any(is.na(train_dataset$x)) #use your notation (train$x or x_train)
any(is.na(train_dataset$y))
any(is.na(test_dataset$x))
any(is.na(test_dataset$y))
19
Find_k = createDataPartition(y_train, p=0.03, list=FALSE, times=1)
train_kx = x_train[Find_k,]
train_ky = y_train[Find_k]
error_train_set <- replicate(0,41)
for(k in 1:41){
  predictions <-knn(train=train_kx, test=train_kx, cl=train_ky,k)
  error_train_set[k] <- 1-mean(predictions==train_ky)
}
error_train_set <- unlist(error_train_set, use.names=FALSE)
error_test_set <- replicate(0,41)
for(k in 1:41){
  predictions <- knn(train=train_kx, test=x_test, cl=train_ky, k)
  error_test_set[k] <- 1-mean(predictions==y_test)
}
error_test_set <- unlist(error_test_set, use.names=FALSE)
png("1.png", height=800, width=1000)
20
plot(error_train_set, type="o", ylim=c(0,1.00), col="blue", xlab="K values",
     ylab="Misclassification errors", main="Test vs train error for varying k values without
PCA")
lines(error_test_set, type="o", col="red")
legend("topright", legend=c("Training error", "Test error"), col=c("blue", "red"), lty=1:1)
dev.off()
# kNN classification optimal k=3
predict_withoutPCA <- knn(train=x_train, test=x_test, cl=y_train, k=3, prob=TRUE)
result_withoutPCA <- cbind(x_test, predict_withoutPCA)
combinetest_withoutPCA <- cbind(x_test, y_test)
# kNN classification optimal k=5
predict_withoutPCA <- knn(train=x_train, test=x_test, cl=y_train, k=5, prob=TRUE)
result_withoutPCA <- cbind(x_test, predict_withoutPCA)
combinetest_withoutPCA <- cbind(x_test, y_test)
# kNN classification optimal k=7
predict_withoutPCA <- knn(train=x_train, test=x_test, cl=y_train, k=7, prob=TRUE)
result_withoutPCA <- cbind(x_test, predict_withoutPCA)
combinetest_withoutPCA <- cbind(x_test, y_test)
# predict_withoutPCA
21
y_test <- factor(y_test, levels = levels(factor(predict_withoutPCA)))
# confusion matrix
cm_withoutPCA <- confusionMatrix(y_test, predict_withoutPCA)
print(cm_withoutPCA)
#####################################################################
min(x_train)
max(x_train)
#covariance matrix
cov_train <- cov(x_train)
pca_train_data <- prcomp(x_train, center=TRUE, scale.=FALSE)
# explained variance
var_explained <- summary(pca_train_data)$importance[2, ]
plot(var_explained, type = 'b', main = "Explained Variance by Principal Components", xlab
     = "Principal Component", ylab = "Proportion of Variance Explained")
22
# Determine the number of components using Heuristics
cum_var_explained <- cumsum(var_explained)
num_components <- which(cum_var_explained >= 0.80)[1] # at least 80% variance
explained
cum_var_explained
# Plot
plot(cum_var_explained, type = "b", xlab = "Number of Principal Components",
     ylab = "Cumulative Proportion of Variance Explained",
     main = "Variance Explained vs Number of Principal Components")
str(pca_train_data)
View(pca_train_data$x)
##########################################################
#Plot PCA with label
tr_label <- y_train
plot(pca_train_data$x, col=tr_label, main = 'PC1 vs PC2 by Label')
plot(pca_train_data$x[,1], col = tr_label, main = 'Variance of Pixels in the Sample for PC
1', ylab = '', xlab
     = 'PC 1')
23
plot(pca_train_data$x[,2],col = tr_label, main = 'over PC 2', ylab = '', xlab = 'PC 2')
plot(pca_train_data$x[,44],col = tr_label, main = 'over PC 44', ylab = '', xlab = 'PC 44')
trainFinal = as.matrix(x_train) %*% pca_train_data$rotation[,1:44]
head(trainFinal)
######################################################################
######################
library(ggplot2)
train_labels <- train_dataset$y
ggplot(trainFinal, aes(PC1, PC2, color = train_labels)) +
  geom_point() +
  labs(x = "PC1", y = "PC2") +
  ggtitle("PCA for 1st and 2nd components")
ggplot(trainFinal, aes(PC1, PC2, color = train_labels)) +
  geom_point() +
  labs(x = "PC1", y = "PC44") +
  24
ggtitle("PCA for 1st and 2nd components")
ggplot(trainFinal, aes(PC1,PC2,color=train_labels)) + geom_point() +
  labs(x= "PC1",y= "PC44") + ggtitle("PCA for 1st and 2nd components")
sc_train <- as.data.frame(scale(x_train, scale = FALSE, center = TRUE))
sc_test <- as.data.frame(scale(x_test, scale = FALSE, center = TRUE))
#PCA on default (normalized and standardized) data
pca_train_data <- prcomp(sc_train)
summary(pca_train_data)
pca_train_data <- prcomp(x_train/255.0, center=TRUE, scale=FALSE)
####### Reconstruction of MNIST digits with different number of PC #############
reconstruction_1PC = t(t(pca_train_data$x[,1:1] %*%
                           t(pca_train_data$rotation[,1:1])) +
                         pca_train_data$center)
reconstruction_44PC = t(t(pca_train_data$x[,1:44] %*%
                            t(pca_train_data$rotation[,1:44])) +
                          pca_train_data$center)
reconstruction_150PC = t(t(pca_train_data$x[,1:150] %*%
                             t(pca_train_data$rotation[,1:150])) +
                           25
                         pca_train_data$center)
reconstruction_784PC = t(t(pca_train_data$x[,1:784] %*%
                             t(pca_train_data$rotation[,1:784])) +
                           pca_train_data$center)
par(mfrow=c(2,2))
show_digit(reconstruction_1PC[400,], main="1 Component") # Reconstruct digit at index
500
show_digit(reconstruction_44PC[400,], main="44 Components")
show_digit(reconstruction_150PC[400,], main="150 Components")
show_digit(reconstruction_784PC[400,], main="784 Components")
############################## KNN ON PCA
######################################
# Apply PCA
pca_train_data <- prcomp(x_train/255.0, center=TRUE, scale=FALSE)
# Select no. of principal components
num_components <- 44
# Transform using selected principal components
26
train_final <- as.matrix(x_train) %*% pca_train_data$rotation[, 1:num_components]
# k =5
k <- 5 # Choose the value of k for kNN classifier
predict_pca <- knn(train = train_final, test = x_test %*% pca_train_data$rotation[,
                                                                                  1:num_components], cl = y_train, k = k)
# confusion matrix
confusion_matrix <- confusionMatrix(factor(predict_pca, levels = levels(factor(y_train))),
                                    factor(y_test))
# Print confusion matrix
print(confusion_matrix)
# scree plot
fviz_eig(pca_train_data, addlabels = TRUE)
########################### Decision Tree ##############
#########################################################
#######################################################
# Decision Tree Classification on Original Data
# Train Decision Tree
dt_model <- rpart(y_train ~ ., data = as.data.frame(x_train), method = "class")
# Visualize Decision Tree
27
rpart.plot(dt_model, main = "Decision Tree Visualization")
# Predictions on Test Data
predictions_dt <- predict(dt_model, as.data.frame(x_test), type = "class")
# Confusion Matrix
cm_dt <- confusionMatrix(factor(predictions_dt), factor(y_test))
print(cm_dt)
# Decision Tree Classification on PCA Data
# Train Decision Tree on PCA-transformed data
dt_model_pca <- rpart(y_train ~ ., data = as.data.frame(train_final), method = "class")
# Visualize Decision Tree on PCA-transformed data
rpart.plot(dt_model_pca, main = "Decision Tree Visualization on PCA-transformed Data")
# Predictions on Test Data using PCA-transformed data
predictions_dt_pca <- predict(dt_model_pca, as.data.frame(x_test %*%
                                                            pca_train_data$rotation[, 1:num_components]), type = "class")
# Confusion Matrix for PCA-transformed data
cm_dt_pca <- confusionMatrix(factor(predictions_dt_pca), factor(y_test))
print(cm_dt_pca)
##################################
###################################
# Decision Tree Classification on Original Data
# Train
decision_tree_ml <- rpart(y_train ~ ., data = as.data.frame(x_train), method = "class")
# Predictions
28
predictions_dt <- predict(decision_tree_ml, as.data.frame(x_test), type = "class")
# Confusion Matrix
confusion_m_decision_tree <- confusionMatrix(factor(predictions_dt), factor(y_test))
print(confusion_m_decision_tree)
# Error Rate
error_rate_dt <- 1 - confusion_m_decision_tree$overall["Accuracy"]
print(paste("Error Rate on Original Data:", error_rate_dt))
# Decision Tree Classification on PCA Data
# Train on PCA-transformed data
decision_tree_ml_pca <- rpart(y_train ~ ., data = as.data.frame(train_final), method =
                                "class")
# Predictions PCA-transformed data
predictions_dt_pca <- predicon(decision_tree_ml_pca, as.data.frame(x_test %*%
                                                                     pca_train_data$rotation[, 1:num_components]), type = "class")
# Confusion Matrix with PCA
confusion_m_decision_tree_pca <- confusionMatrix(factor(predictions_dt_pca),
                                                 factor(y_test))
print(confusion_m_decision_tree_pca)
# Error Rate for PCA
error_rate_dt_pca <- 1 - confusion_m_decision_tree_pca$overall["Accuracy"]
print(paste("Error Rate on PCA-transformed Data:", error_rate_dt_pca))
##################################
#####################################
error_rate_knn <- 1-0.9695
29
# Error rate for decision tree without PCA
error_rate_dt <- 1 - confusion_m_decision_tree$overall["Accuracy"]
# Error rate comparison
error_rates <- c(error_rate_knn, error_rate_dt)
methods <- c("kNN", "Decision Tree")
library(ggplot2)
# Assuming 'error_rates' is a vector of error rates and 'methods' is a vector of method
names
df <- data.frame(Method = methods, ErrorRate = error_rates)
# Create the bar plot
ggplot(df, aes(x = Method, y = ErrorRate, fill = Method)) +
  geom_bar(stat = "identity") +
  labs(title = "Error Rate Comparison", y = "Error Rate") +
  scale_fill_manual(values = c( "#ea7a87" , "#ea273a")) +
  theme_minimal()
################
##################
#################
# decision tree visualization
library(rpart.plot)
rpart.plot(decision_tree_ml, main = "Decision Tree Visualization")
rpart.plot(decision_tree_ml, main = "Decision Tree Visualization", box.palette = "Paired",
           cex = 0.8)
plot(decision_tree_ml, uniform = TRUE, main = "Decision Tree for MNIST Data")
30
text(decision_tree_ml, use.n = TRUE, all = TRUE, cex = .8)