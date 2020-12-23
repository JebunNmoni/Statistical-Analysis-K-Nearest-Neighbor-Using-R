############### INITIALIZATION #################
# Clear console
cat("\014")

# Remove all variables
rm(list=ls(all=TRUE)) 

# Set working directory
setwd('FilePath)

# ISLR: Data for an Introduction to Statistical Learning with Applications in R
# Provides the collection of data-sets used in the book 'An Introduction to Statistical Learning with Applications in R'.
library(ISLR)

#class: Functions for Classification
# Various functions for classification, including k-nearest neighbour, Learning Vector Quantization and Self-Organizing Maps.
library(class)

############### LOAD DATA #################
# Read csv files
testData=read.csv("1-test_data.csv",header = T)
trainingData=read.csv("1-training_data.csv",header=T)
# Return the first or last parts of a vector, matrix, table, data frame or function using head() function
head(trainingData)
head(testData)
# Compactly display the internal structure of an R object using str() function
str(trainingData)
str(testData)

############### CREATE RESPONSE AND PREDICTOR OBJECTS #################
# cbind Takes a sequence of vector, matrix or data-frame arguments and combine by columns or rows, respectively.
trainingData.x=cbind(trainingData$x.1,trainingData$x.2)
trainingData.y=trainingData$y
testData.x=cbind(testData$x.1,testData$x.2)  
testData.y=testData$y

plot(trainingData.x, col = ifelse(trainingData.y == "yes", "green", "red"), pch= ifelse(trainingData.y == "yes", 1, 20),xlab="Training Data x1",ylab="Training Data x2",
     main=paste("Training Data"))

############### INITIALIZE K AND TRAINING AND TESTING ERROR RATE #################
k=c(seq(1,30,by=1),seq(35,100,by=5))
n=length(k)
# Training error rate
trainingData.error.rate = numeric(length = n)
# Testing error rate
testData.error.rate = numeric(length = n)
# names: Functions to get or set the names of an object.
names(trainingData.error.rate) = names(testData.error.rate) = k

############### APPLY KNN METHOD #################
# knn(train, test, cl, k = 1, l = 0, prob = FALSE, use.all = TRUE)
# -- Arguments --
# train: matrix or data frame of training set cases.
# test:matrix or data frame of test set cases. A vector will be interpreted as a row vector for a single case.
# cl: factor of true classifications of training set
# k: number of neighbours considered.
# l: minimum vote for definite decision, otherwise doubt. (More precisely, less than k-l dissenting votes are allowed, even if k is increased by ties.)
# prob: If this is true, the proportion of the votes for the winning class are returned as attribute prob.
# use.all: controls handling of ties. If true, all distances equal to the kth largest are included. If false, a random selection of distances equal to the kth is chosen to use exactly k neighbours.
for (i in seq(along = k)) {
  # Set the seed of R's random number generator, which is useful for creating simulations or random objects that can be reproduced.
  set.seed(1)
  # k-nearest neighbour classification for training set.
  mod.trainingData = knn(trainingData.x, trainingData.x, trainingData.y, k = k[i])
  # Training error rate
  trainingData.error.rate[i] = 1 - sum(mod.trainingData == trainingData.y)/length(trainingData.y)
  set.seed(1)
  # k-nearest neighbour classification for test set.
  mod.testData = knn(trainingData.x, testData.x, trainingData.y, k = k[i])
  # Test error rate
  testData.error.rate[i] = 1 - sum(mod.testData == testData.y)/length(testData.y)
}

# Create objects with error rate
result = data.frame(k, trainingData.error.rate, testData.error.rate)

# Plot training and test error rates with respect to k
plot(k, trainingData.error.rate, xlab = "Number of nearest neighbors", ylab = "Error rate", 
     main="Plot of training and test error rates against k", 
     type = "b", ylim = range(c(trainingData.error.rate, testData.error.rate)), col = "black", pch = 20)
lines(k, testData.error.rate, type="b", col="red", pch = 1)
legend(40,0.05, lty = 1, col = c("black", "red"), pch=c(20,1),legend = c("Training Error Rate", "Test Error Rate"),
       cex=1)

# Get optimal value of k. We used the least flexible model (i.e the highest k)
opt=result[testData.error.rate == min(result$testData.error.rate), ]

####### Decision boundary for optimal k with training data #######

# Set up the grid
n.grid = 50
x1.grid = seq(min(trainingData.x[, 1]), max(trainingData.x[, 1]), length.out = n.grid)
x2.grid = seq(min(trainingData.x[, 2]), max(trainingData.x[, 2]), length.out = n.grid)
grid = expand.grid(x1.grid, x2.grid)

# Apply knn for optimal k on grid
k.opt = max(opt$k)
set.seed(1)
mod.opt = knn(trainingData.x, grid, trainingData.y, k.opt, prob = T)
# attr: Get or set specific attributes of an object.
prob = attr(mod.opt, "prob") # prob is voting fraction for winning class, i.e. the probability of winning.
prob = ifelse(mod.opt == "yes", prob, 1 - prob) # now it is voting fraction for Decision == "Yes"
prob = matrix(prob, n.grid, n.grid)

# Plot training data and decision boundary for optimal k
abline(v=k.opt, col="blue")
legend(20,0.1, lty = 1, col = c("black", "red"), pch=c(20,1),legend = c(paste("Optimum Training Error Rate = ",opt$trainingData.error.rate), paste("Optimum Test Error Rate",opt$testData.error.rate)),
       cex=1)
plot(trainingData.x, col = ifelse(trainingData.y == "yes", "green", "red"), pch= ifelse(trainingData.y == "yes", 1, 20),xlab="Training Data x1",ylab="Training Data x2",
     main=paste("Decision Boundary for Optimal k\nk=", k.opt))
contour(x1.grid, x2.grid, prob, levels = 0.5, add = T)

