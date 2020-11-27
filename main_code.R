##############################
# installing required packages
##############################

if(!require(tidyverse))    install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret))        install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(ggthemes))     install.packages("ggthemes", repos = "http://cran.us.r-project.org")
if(!require(ggcorrplot))   install.packages("ggcorrplot", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(magrittr))     install.packages("magrittr", repos = "http://cran.us.r-project.org")
if(!require(rpart))        install.packages("rpart", repos = "http://cran.us.r-project.org")
if(!require(rpart.plot))   install.packages("rpart.plot", repos = "http://cran.us.r-project.org")
if(!require(neighbr))      install.packages("neighbr", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(ggthemes)
library(ggcorrplot)
library(randomForest)
library(magrittr)
library(rpart)
library(rpart.plot)
library(neighbr)




#########################
# import and wrangle data
#########################

# the github repo with the data set "diabetes_data_upload" is available here: https://github.com/alyomahoney/Diabetes_Project 
# the file "diabetes_data_upload.csv" provided in the github repo must be included in the working (project) directory for the code below to run

# read in the data and replace spaces in column names with full stops
data <- read_csv("diabetes_data_upload.csv", col_types = "dffffffffffffffff")
colnames(data) <- make.names(colnames(data))

# reorder the "yes-no" factors so that all are in the same order
# when reading in the data, if the first entry of a column is "No" for example, that is taken to be the first level
# this loop changes the order of the factors for which the first entry is "No"
no_ind <- which(data[1,]=="No")
for (i in no_ind) {
  data[,i] <- factor(data[[i]], levels = c("Yes","No"))
}

# create a validation set - this is used to assess the final model
# diabetes data set is used for model training and selection
set.seed(4)
validation_index <- createDataPartition(data$class, times = 1, p = 0.15, list = FALSE)
validation <- data %>%
  slice(validation_index)
diabetes <- data %>%
  slice(-validation_index)

# create train and test sets from diabetes. train is used to construct various models and test is used to assess their performances
# the best performing model will then be retrained using the diabetes data set and assessed using the validation data set
set.seed(16)
test_index <- createDataPartition(diabetes$class, times = 1, p = 0.15, list = FALSE)
train <- diabetes %>% slice(-test_index)
test <- diabetes %>% slice(test_index)




##################
# data exploration
##################

# set global theme (google docs)
theme_set(theme_gdocs())

# lets first visualise the corrrelation between the features (non significant correlations left blank)
# probably not worth using pca because the data set is quite small
bin_diabetes <- sapply({diabetes[,c(-1,-2,-17)] == "Yes"} %>% as_tibble, as.numeric) %>% as_tibble %>% mutate(Gender = as.numeric(diabetes$Gender=="Male"))
cor_diabetes <- bin_diabetes %>% cor
p.cor_diabetes <- bin_diabetes %>% cor_pmat
cor_diabetes %>%
  ggcorrplot(lab = TRUE, type = "lower", method = "circle",
             insig = "blank", p.mat = p.cor_diabetes,ggtheme = theme_gdocs(),
             colors = c("#6D9EC1", "white", "#E46726"),
             title = "Correlation Plot of variables in Diabetes Data Set", legend.title = "Correlation")
ggsave("rmd_files/images/correlation.png", width = 9, height = 9)

# compare genders. not many non-diabetic patients are female
diabetes %>%
  ggplot(aes(class, fill = Gender)) +
  geom_bar(width = 0.6, position = position_dodge(width = 0.7)) +
  ylab("Number of Patients") +
  ggtitle("Number of Diabetic and Non-Diabet Patients by Gender")
ggsave("rmd_files/images/gender.png", width = 9, height = 7)

# now comparing age. there doesn't seem to be a significant difference
diabetes %>%
  ggplot(aes(Age, class, fill = class)) +
  geom_violin(alpha = 0.8) +
  ggtitle("Prevalence of Diabetes by Age")
ggsave("rmd_files/images/age.png", width = 9, height = 7)

# polyuria and polydipsia appear to be significant
# if a patient has both conditions, they are very likely to be diabetic
diabetes %>%
  ggplot(aes(Polyuria, Polydipsia, colour = class)) +
  geom_jitter(height = 0.2, width = 0.2) +
  ggtitle("Prevalence of Diabetes by Polydipsia and Polyuria")
ggsave("rmd_files/images/dipsiauria.png", width = 9, height = 7)

# this plot suggests that weakness might not be a significant feature
# however, weight loss could be significant
diabetes %>%
  ggplot(aes(sudden.weight.loss, weakness, colour = class)) +
  geom_jitter(height = 0.2, width = 0.2) +
  xlab("Sudden Weight Loss") +
  ylab("Weakness") +
  ggtitle("Prevalence of Diabetes by Sudden Weight Loss and Weakness")
ggsave("rmd_files/images/weightweak.png", width = 9, height = 7)

# if a patient has both visual blurring and genital thrush then this plot indicates that they are probably diabetic
diabetes %>%
  ggplot(aes(Genital.thrush, visual.blurring, colour = class)) +
  geom_jitter(height = 0.2, width = 0.2) +
  xlab("Genital Thrush") +
  ylab("Visual Blurring") +
  ggtitle("Prevalence of Diabetes by Genital Thrush and Visual blurring")
ggsave("rmd_files/images/thrushblurring.png", width = 9, height = 7)

# this is a very interesting plot - it suggests that obesity isn't a really significant factor
# this goes against general expectations
diabetes %>%
  ggplot(aes(class, fill = Obesity)) +
  geom_bar(width = 0.6, position = position_dodge(width = 0.7)) +
  ggtitle("Prevalence of Diabetes by Obesity") +
  ylab("Number of Patients")
ggsave("rmd_files/images/obesity.png", width = 9, height = 7)




#####################
# logistic regression
#####################

# this section constructs a logistic regression model
# the cutoff is chosen so that the mean of the accuracy and the sensitivity is maximised
# sensitivity is an important measure to consider since not identifying a diabetic person is more costly than incorrectly classifying someone who is non-diabetic

# the glm returns the probability that a given patient is diabetic
# to turn this into predictions, typically p would be set to 0.5 and and predictions greater than p would be classed as Positive, and Negative otherwise
# since sensitivity is important, it might be of interest for p to be lower than 0.5

# 5 fold cross-validation is used to select an optimal p
# an optimal p is one which maximises the mean of the accuracy and the sensitivity
tune_p <- seq(0.2, 0.5, by = 0.01)                   # cutoff values to try
seed_mat <- matrix(nrow = 10, ncol = length(tune_p)) # empty matrix which will contain results from all cross-validations
folds <- 5                                           # number of folds in cv
acc_p <- matrix(nrow = folds, ncol = length(tune_p)) # empty matrix which will contain results from each cv

# setting different seeds gives drastically different results, so 10 different seeds are set, and cv is performed for each of them
# for each seed, normal cv is carried out and the mean result (mean of accuracy and sensitivity for each p) is stored in seed_mat
for (j in 1:10) {
  
  # index to split the train data set
  set.seed(j)
  ind <- createFolds(1:nrow(train), k = folds)
  
  for (i in 1:folds) {
    
    # create train and test sets for cv
    cv_train <- train %>% slice(-ind[[i]])
    cv_test <- train %>% slice(ind[[i]])
    
    # fill matrix with results (mean of accuracy and sensitivity) from glm with cutoff p
    acc_p[i,] <- sapply(tune_p, function(p){
      
      # create the glm
      cv_mod_glm <- glm(as.numeric(class=="Positive")~., family = "binomial", data = cv_train)
      
      # obtain the predictions (these are probabilities)
      cv_preds_glm <- predict(cv_mod_glm, cv_test, type = "response")
      
      # if prediction>p, class as Positive. 
      cv_cm <- confusionMatrix(ifelse(cv_preds_glm>p, "Positive","Negative") %>% factor(levels = c("Positive","Negative")), cv_test$class)
      
      # return the mean of the accuracy and sensitivity
      return(mean(c(cv_cm$overall["Accuracy"], cv_cm$byClass["Sensitivity"])))
    })
    
  }
  # populate seed_mat with the mean results for each seed
  seed_mat[j,] <- colMeans(acc_p)
  
  # keep track of how many seeds have been run
  cat("seed",j,"out of 10 complete\n")
}

# define the value of p which maximeses the mean of the accuracy and sensitivity
opt_p <- tune_p[which(min_rank(desc(colMeans(seed_mat)))==1)]

# data frame which is fed into ggplot (below)
glm_cv_dat <- tibble(p = tune_p,
                     acc = colMeans(seed_mat))

# visualise the results from cv
glm_cv_dat %>%
  ggplot(aes(tune_p, acc)) +
  geom_point() +
  geom_point(aes(opt_p, max(acc)), shape = 5, size = 5) +
  xlab("Cutoff (p)") +
  ylab("Mean of Accuracy and Sensitivity") +
  ggtitle("Mean of Accuracy and Sensitivity for Various Cutoffs")
ggsave("rmd_files/images/cv_p.png", width = 8, height = 5)

# now use the entire train data set and evaluate the model against the test set
model_glm <- glm(as.numeric(class=="Positive")~., family = "binomial", data = train)
preds_glm <- predict(model_glm, test, type = "response")                                                  # these are probabilities
preds_glm <- ifelse(preds_glm>opt_p, "Positive","Negative") %>% factor(levels = c("Positive","Negative")) # convert probabilities into factors (Positive or Negative)

# confusion matrix
cm_glm <- confusionMatrix(preds_glm, test$class)

# save accuracy and sensitivity
acc_glm <- cm_glm$overall["Accuracy"]
sen_glm <- cm_glm$byClass["Sensitivity"]




#######################
# k-nearest neighrbours
#######################

# this section constructs a knn model using the jaccard distance metric (https://people.revoledu.com/kardi/tutorial/Similarity/BinaryVariables.html)

# the knn function in the neighbr package requires the features to be binary (logical or 0-1 numeric) when using the jaccard distance
# all of the features are binary apart from age, so it is ommited from this model
# the train data set is modified so that the features are logical
# a column ID is also required to run the knn function
train_knn <- {train[-c(1, 2, 17)]=="Yes"} %>% as_tibble # remove age, gender and class. then, convert to logical
train_knn <- cbind(train[2] == "Male", train_knn)       # add logical variable for gender
train_knn <- cbind(train_knn, train[17])                # add the class back in (it doesn't need to be logical, only the features)
train_knn$ID <- 1:nrow(train_knn)

# the test set needs to take the same format
# however, the ID and class columns need removed (requirement of knn function)
test_knn <- {test[-c(1, 2, 17)]=="Yes"} %>% as_tibble
test_knn <- cbind(test[2] == "Male", test_knn)

# here, 5-fold cv is used to select an optimal k
tune_k <- seq(3, 9, by = 2)                              # values of k to try
folds <- 5                                               # number of folds in cv
acc_k <- matrix(nrow = folds, ncol = length(tune_k))     # empty matrix which will contain results from each cv
set.seed(4)
ind <- createFolds(1:nrow(train), k = folds)             # index to split train_knn into folds for cv

# this cv returns the mean of the accuracy and sensitivity, not just the accuracy
for (i in 1:folds) {
  
  # create train and test sets for cv
  cv_train <- train_knn %>% slice(-ind[[i]])
  cv_test <- train_knn %>% slice(ind[[i]]) %>% select(-c("class","ID")) # requirement of knn function
  
  # fill matrix with accuracy results from glm model with cutoff p
  acc_k[i,] <- sapply(tune_k, function(k){
    
    # create the knn model using jaccard distance metric
    cv_mod_knn <- knn(train_set = cv_train,
                      test_set = cv_test,
                      k = k,
                      categorical_target = "class",
                      comparison_measure="jaccard",
                      id="ID")
    
    # define predictions for the knn model
    cv_preds_knn <- cv_mod_knn$test_set_scores$categorical_target %>% factor(levels = c("Positive","Negative"))
    
    # create the confusion matrix
    cv_cm <- confusionMatrix(cv_preds_knn, train_knn %>% slice(ind[[i]]) %>% .$class)
    
    # return the mean of the accuracy and sensitivity
    return(mean(c(cv_cm$overall["Accuracy"], cv_cm$byClass["Sensitivity"])))
    
  })
  # keep track of how many folds have been run
  cat("fold",i,"out of",folds,"complete\n")
}

# define the optimal k
opt_k <- tune_k[which.max(colMeans(acc_k))]

# data frame which is fed into ggplot (below)
knn_cv_dat <- tibble(k = tune_k,
                     acc = colMeans(acc_k))

# visualise the cv results
knn_cv_dat %>%
  ggplot(aes(tune_k, acc)) +
  geom_point() +
  geom_point(aes(opt_k, max(acc)), shape = 5, size = 5) +
  xlab("k") +
  ylab("Mean of Accuracy and Sensitivity") +
  ggtitle("Mean of Accuracy and Sensitivity for Various Values of k")
ggsave("rmd_files/images/cv_k.png", width = 8, height = 5)

# retrain the knn model on train_knn
model_knn <- knn(train_set = train_knn,
                 test_set = test_knn,
                 k = opt_k,
                 categorical_target = "class",
                 comparison_measure="jaccard",
                 id="ID")

# get predictions for knn_test using the model 
preds_knn <- model_knn$test_set_scores$categorical_target %>% factor(levels = c("Positive","Negative"))

# confusion matrix
cm_knn <- confusionMatrix(preds_knn, test$class)

# save accuracy and sensitivity
acc_knn <- cm_knn$overall["Accuracy"]
sen_knn <- cm_knn$byClass["Sensitivity"]




###############
# decision tree
###############

# this section constructs a decision tree
# this is probably the most interpretable model, it's easy to visualise how it makes decisions

# the train function in the caret package is used to select an optimal complexity parameter (cp) using 25 bootstrap samples with replacement
set.seed(64)
model_tree_cv <- train(class~.,
                       method = "rpart",
                       tuneGrid = data.frame(cp = seq(0, 0.05, len = 25)),
                       data = diabetes)

# visualise the performance of each cp
ggplot(model_tree_cv, highlight = TRUE)
ggsave("rmd_files/images/cv_cp_tree.png", width = 8, height = 5)

# observe the large error for each cp
model_tree_cv$results %>% 
  ggplot(aes(x = cp, y = Accuracy)) +
  geom_line() +
  geom_point() +
  geom_errorbar(aes(x = cp, 
                    ymin = Accuracy - AccuracySD,
                    ymax = Accuracy + AccuracySD))

# define the optimal cp
opt_cp <- model_tree_cv$bestTune

# redefine the model using the train data set and optimal cp
model_tree <- rpart(class~., cp = opt_cp, data = train) 

# plot the model - this really helps to understand how the algorithm works
rpart.plot(model_tree, type = 5)
title("Decision Tree")

# the predict function returns probabilities, much like the logistic regression model
# these probabilities are are the proportions of each class that exists in each node
# again, a typical cutoff is 0.5, however sensitivity is important in this scenario
# cv will decide on an optimal cutoff which increases the mean of the accuracy and the sensitivity

# here, 5-fold cv is used to select an optimal c
tune_co <- seq(0.1, 0.8, by = 0.025)                     # values of cutoff (co) to try
folds <- 5                                               # number of folds in cv 
acc_co <- matrix(nrow = folds, ncol = length(tune_co))   # empty matrix which will contain results from each cv
set.seed(4)
ind <- createFolds(1:nrow(train), k = folds)             # index to split train_knn into folds for cv

# this cv returns the mean of the accuracy and sensitivity, not just the accuracy
for (i in 1:folds) {
  
  # create train and test sets for cv
  cv_train <- train %>% slice(-ind[[i]])
  cv_test <- train %>% slice(ind[[i]])
  
  # fill matrix with accuracy results from glm model with cutoff p
  acc_co[i,] <- sapply(tune_co, function(co){
    
    # create the knn model using jaccard distance metric
    cv_mod_tree <- rpart(class~., cp = opt_cp, data = cv_train)
    
    # define predictions for the knn model
    cv_preds_tree <- predict(cv_mod_tree, cv_test) %>%
      as_tibble %$%
      ifelse(Positive > co, "Positive", "Negative") %>% 
      factor(levels = c("Positive", "Negative"))
    
    # create the confusion matrix
    cv_cm <- confusionMatrix(cv_preds_tree, cv_test$class)
    
    # return the mean of the accuracy and sensitivity
    return(mean(c(cv_cm$overall["Accuracy"], cv_cm$byClass["Sensitivity"])))
    
  })
  # keep track of how many folds have been run
  cat("fold",i,"complete\n")
}

# define the optimal value for k
opt_co <- tune_co[which.max(colMeans(acc_co))]

# data frame which is fed into ggplot (below)
tree_cv_dat <- tibble(co = tune_co,
                      acc = colMeans(acc_co))

# visualise the cv results
tree_cv_dat %>%
  ggplot(aes(tune_co, acc)) +
  geom_point() +
  geom_point(aes(opt_co, max(acc)), shape = 5, size = 5) +
  xlab("p") +
  ylab("Mean of Accuracy and Sensitivity") +
  ggtitle("Mean of Accuracy and Sensitivity for Various Values of co")

# there are lots of values which achieve the same maximum result, so the median of the cutoffs is used
opt_co <- median(tune_co[min_rank(desc(colMeans(acc_co)))==1])

# visualise the cv results with the new opt_co
tree_cv_dat %>%
  ggplot(aes(tune_co, acc)) +
  geom_point() +
  geom_point(aes(opt_co, max(acc)), shape = 5, size = 5) +
  xlab("p") +
  ylab("Mean of Accuracy and Sensitivity") +
  ggtitle("Mean of Accuracy and Sensitivity for Various Values of p")
ggsave("rmd_files/images/cv_co.png", width = 8, height = 5)

# obtain predictions using opt_p
preds_tree <- predict(model_tree, validation) %>%
  as_tibble %$%
  ifelse(Positive > opt_co, "Positive", "Negative") %>% 
  factor(levels = c("Positive", "Negative"))

# confusion matrix
cm_tree <- confusionMatrix(preds_tree, validation$class)

# save accuracy and sensitivity
acc_tree <- cm_tree$overall["Accuracy"]
sen_tree <- cm_tree$byClass["Sensitivity"]




###############
# random forest
###############

# this section expands on the idea of decision trees by creating a random forest
# a random forest is a collection of decision trees
# predictions are made using the majority votes from each tree
# to reduce the dependency between trees, a random subset of features can be chosen at each node to decide on which split to make (if any). this is `mtry`

# one disadvantage of random forests is that they lose interpretability (in comparison to decision trees)
# the algorithm isn't easy to visualise. it is probably best to think of it just as an extension of decision trees

# the train function in the caret package is used to choose an optimal mtry
# note that the cutoff is opt_co, calculated in the decision tree model. this accounts for the fact that sensitivity is important
# the default is to construct 500 trees
set.seed(11)
model_rf <- train(class~.,
                  method = "rf",
                  tuneGrid = data.frame(mtry = 3:11),
                  cutoff = c(opt_co,1-opt_co),
                  data = train)

# visualise the performance of each mtry
ggplot(model_rf, highlight = TRUE) +
  scale_x_discrete(limits = 2:12) +
  ggtitle("Accuracy for each number of randomly selected predictors")
ggsave("rmd_files/images/mtry.png", width = 8, height = 5)

# again, note the variability of each mtry
model_rf$results %>% 
  ggplot(aes(x = mtry, y = Accuracy)) +
  geom_line() +
  geom_point() +
  geom_errorbar(aes(x = mtry, 
                    ymin = Accuracy - AccuracySD,
                    ymax = Accuracy + AccuracySD))

# create predictions
preds_rf <- predict(model_rf, test)

# confusion matrix
cm_rf <- confusionMatrix(preds_rf, test$class)

# the importance of each variable is also accessible via the importance function
# polyuria is a clear winner, meaning it is likely to be the root note in most of the decision trees in the forest
importance(model_rf$finalModel)

# save accuracy and sensitivity
acc_rf <- cm_rf$overall["Accuracy"]
sen_rf <- cm_rf$byClass["Sensitivity"]




##########
# ensemble
##########

# this section creates an ensemble using the three best performing models
# since the decision tree performed the worst, it is dropped.
# it makes sense to drop the decision tree model since the random forest model is essentially an improved version

# store the predictions from the three models in a data frame
all_preds <- tibble(glm = preds_glm,
                    knn = preds_knn,
                    rf  = preds_rf)

# the predictions of the ensemble are obtained by majority votes
preds_ens <- apply(all_preds,1,function(x) names(which.max(table(x)))) %>%
  factor(levels = c("Positive","Negative"))

# confusion matrix (it actually performs worse than the random forest)
cm_ens <- confusionMatrix(preds_ens, test$class)

# save accuracy and sensitivity
acc_ens <- cm_ens$overall["Accuracy"]
sen_ens <- cm_ens$byClass["Sensitivity"]

# store all results (rmd)
results <- tibble(Method = c("Logistic Regression","k-Nearest Neighbours","Decision Tree","Random Forest","Ensemble"),
                  Accuracy = c(acc_glm, acc_knn, acc_tree, acc_rf, acc_ens),
                  Sensitivity = c(sen_glm, sen_knn, sen_tree, sen_rf, sen_ens)) %>%
  mutate(Mean = rowMeans(select(., Accuracy, Sensitivity)))




#############################
# final model - random forest
#############################

# the random forest had the highest accuracy and sensitivity out of all of the models, including the ensemble
# therefore, it is chosen to be the final model and is reconstructed using the diabetes data set

# training the model (using opt_co calculated in the decision tree model)
# again, cv is used to choose an optimal mtry (25 bootstrap samples with replacement)
set.seed(1)
final_model_rf <- train(class~.,
                        method = "rf",
                        tuneGrid = data.frame(mtry = 3:11),
                        cutoff = c(opt_co,1-opt_co),
                        data = diabetes)

# visualise the performance of each mtry
ggplot(final_model_rf, highlight = TRUE) +
  scale_x_discrete(limits = 2:12) +
  ggtitle("Accuracy for each number of randomly selected predictors")
ggsave("rmd_files/images/mtry_final.png", width = 8, height = 5)

# optimal mtry is 5
final_model_rf$bestTune

# defining the predictions
final_preds_rf <- predict(final_model_rf, validation)

cm_final <- confusionMatrix(final_preds_rf, validation$class)

# importance of each feature
imp_final <- importance(final_model_rf$finalModel)

# save accuracy and sensitivity
acc_final <- cm_final$overall["Accuracy"]
sen_final <- cm_final$byClass["Sensitivity"]
