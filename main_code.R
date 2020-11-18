# read in the data and replace spaces in column names with decimals
data <- read_csv("diabetes_data_upload.csv", col_types = "dffffffffffffffff")
colnames(data) <- make.names(colnames(data))

# reorder the factors so that all are in the same order
no_ind <- which(data[1,]=="No")
for (i in 1:length(no_ind)) {
  j <- no_ind[i]
  data[,j] <- factor(data[[j]], levels = c("Yes","No"))
}

# create a validation set - this is used to assess the final model
set.seed(4)
validation_index <- createDataPartition(data$class, times = 1, p = 0.15, list = FALSE)
validation <- data %>%
  slice(validation_index)
diabetes <- data %>%
  slice(-validation_index)

# create train and test sets. train is used to construct various models and test is used to compare their performances
set.seed(16)
test_index <- createDataPartition(diabetes$class, times = 1, p = 0.15, list = FALSE)
train <- diabetes %>% slice(-test_index)
test <- diabetes %>% slice(test_index)




















theme_set(theme_gdocs())

# compare genders
table(diabetes %>% select(Gender, class))
diabetes %>%
  ggplot(aes(class, fill = Gender)) +
  geom_bar(width = 0.6, position = position_dodge(width = 0.7))

# there isn't really a difference with age
diabetes %>%
  ggplot(aes(Age, class, fill = class)) +
  geom_violin(alpha = 0.8)


# polyuria and polydipsia
diabetes %>%
  ggplot(aes(class, fill = Polyuria)) +
  geom_bar(width = 0.6, position = position_dodge(width = 0.7))

diabetes %>%
  ggplot(aes(class, fill = Polydipsia)) +
  geom_bar(width = 0.6, position = position_dodge(width = 0.7))

diabetes %>%
  ggplot(aes(Polyuria, Polydipsia, colour = class)) +
  geom_jitter(height = 0.2, width = 0.2) # both are good for prediction




bin_diabetes <- sapply({diabetes[,c(-1,-2,-17)] == "Yes"} %>% as_tibble, as.numeric) %>% as_tibble %>% mutate(Gender = as.numeric(diabetes$Gender=="Male"))
cor_diabetes <- bin_diabetes %>% cor
p.cor_diabetes <- bin_diabetes %>% cor_pmat
cor_diabetes %>%
  ggcorrplot(lab = TRUE, type = "lower", method = "circle",
             insig = "blank", p.mat = p.cor_diabetes,ggtheme = theme_gdocs(),
             colors = c("#6D9EC1", "white", "#E46726"),
             title = "Correlation Plot of variables in Diabetes Data Set", legend.title = "Correlation")
# maybe not bother with pca because the data set isn't that big. gdocs might not be ideal because of the black lines


diabetes %>%
  ggplot(aes(sudden.weight.loss, weakness, colour = class)) +
  geom_jitter(height = 0.2, width = 0.2) # weakness not useful, weight loss useful


diabetes %>%
  ggplot(aes(Genital.thrush, visual.blurring, colour = class)) +
  geom_jitter(height = 0.2, width = 0.2) # if you have both of these things


diabetes %>%
  ggplot(aes(Itching, Irritability, colour = class)) +
  geom_jitter(height = 0.2, width = 0.2) # itching useless, irritability useful



diabetes %>%
  ggplot(aes(delayed.healing, partial.paresis, colour = class)) +
  geom_jitter(height = 0.2, width = 0.2) # partial p useful, delayed healing useless


diabetes %>%
  ggplot(aes(muscle.stiffness, Alopecia, colour = class)) +
  geom_jitter(height = 0.2, width = 0.2) # ms yes implies alopecia info is useful


diabetes %>%
  ggplot(aes(Obesity, Polyphagia, colour = class)) +
  geom_jitter(height = 0.2, width = 0.2) # polyphagia useful is not obese

diabetes %>%
  ggplot(aes(class, fill = Obesity)) +
  geom_bar(width = 0.6, position = position_dodge(width = 0.7)) # is obesity maybe not that much of a factor?


















# the coefficients are MLEs

# maybe choose the cutoff with cross validation?
tune_p <- seq(0.05, 0.3, by = 0.01)                   # cutoff values to try
seed_mat <- matrix(nrow = 10, ncol = length(tune_p)) # empty matrix which will contain results from all cross-validations
k <- 5                                               # number of folds in cv
acc_p <- matrix(nrow = k, ncol = length(tune_p))     # empty matrix which will contain results from each cv
for (j in 1:10) {
  
  set.seed(j)
  ind <- createFolds(1:nrow(train), k = k)
  
  for (i in 1:k) {
    
    # create train and test sets for cv
    cv_train <- train %>% slice(ind[[i]])
    cv_test <- train %>% slice(-ind[[i]])
    
    # fill matrix with accuracy results from glm model with cutoff p
    acc_p[i,] <- sapply(tune_p, function(p){
      cv_mod_glm <- glm(as.numeric(class=="Positive")~., family = "binomial", data = cv_train)
      cv_preds_glm <- predict(cv_mod_glm, cv_test, type = "response")
      cv_cm <- confusionMatrix(ifelse(cv_preds_glm>p, "Positive","Negative") %>% factor(levels = c("Positive","Negative")), cv_test$class)
      return(mean(c(cv_cm$overall["Accuracy"], cv_cm$byClass["Sensitivity"])))
    })
    
  }
  
  seed_mat[j,] <- colMeans(acc_p)
  rm(cv_test, cv_train)
}

opt_p <- tune_p[which(min_rank(desc(colMeans(seed_mat)))==1)]
cv_dat <- tibble(p = tune_p,
                 acc = colMeans(seed_mat))
cv_dat %>%
  ggplot(aes(tune_p, acc)) +
  geom_point() +
  geom_point(aes(opt_p, max(acc)), shape = 5, size = 5) +
  xlab("Cutoff") +
  ylab("Mean of Accuracy and Sensitivity") +
  ggtitle("Mean of Accuracy and Sensitivity for Various Cutoffs")





# now use the entire train data set and evaluate the model against the test set
model_glm <- glm(as.numeric(class=="Positive")~., family = "binomial", data = train)
preds_glm <- predict(model_glm, test, type = "response")
preds_glm <- ifelse(preds_glm>opt_p, "Positive","Negative") %>% factor(levels = c("Positive","Negative"))
confusionMatrix(preds_glm, test$class) # experiment with different 0.5 for change in specificity/sensitivity


















train_knn <- {train[-c(1, 2, 17)]=="Yes"} %>% as_tibble
train_knn <- cbind(train[2] == "Male", train_knn)
train_knn <- cbind(train_knn, train[17])
train_knn$ID <- 1:nrow(train_knn)

test_knn <- {test[-c(1, 2, 17)]=="Yes"} %>% as_tibble
test_knn <- cbind(test[2] == "Male", test_knn)

tune_k <- seq(3, 7, by = 2)                              # cutoff values to try
folds <- 5                                               # number of folds in cv (changed to avoid confusion with knn)
acc_k <- matrix(nrow = folds, ncol = length(tune_k))     # empty matrix which will contain results from each cv
set.seed(4)
ind <- createFolds(1:nrow(train), k = folds)

for (i in 1:folds) {
  
  # create train and test sets for cv
  cv_train <- train_knn %>% slice(ind[[i]])
  cv_test <- train_knn %>% slice(-ind[[i]]) %>% select(-c("class","ID"))
  
  # fill matrix with accuracy results from glm model with cutoff p
  acc_k[i,] <- sapply(tune_k, function(k){
    cv_mod_knn <- knn(train_set = cv_train,
                      test_set = cv_test,
                      k = k,
                      categorical_target = "class",
                      comparison_measure="jaccard",
                      id="ID")
    cv_preds_knn <- cv_mod_knn$test_set_scores$categorical_target %>% factor(levels = c("Positive","Negative"))
    cv_cm <- confusionMatrix(cv_preds_knn, train_knn %>% slice(-ind[[i]]) %>% .$class)
    return(mean(c(cv_cm$overall["Accuracy"], cv_cm$byClass["Sensitivity"])))
  })
  cat("fold",i,"complete\n")
}

opt_k <- tune_k[which.max(colMeans(acc_k))]
cv_dat <- tibble(k = tune_k,
                 acc = colMeans(acc_k))
cv_dat %>%
  ggplot(aes(tune_k, acc)) +
  geom_point() +
  geom_point(aes(opt_k, max(acc)), shape = 5, size = 5) +
  xlab("k") +
  ylab("Mean of Accuracy and Sensitivity") +
  ggtitle("Mean of Accuracy and Sensitivity for Various Values of k")

model_knn <- knn(train_set = train_knn,
                 test_set = test_knn,
                 k = 3,
                 categorical_target = "class",
                 comparison_measure="jaccard",
                 id="ID")

preds_knn <- model_knn$test_set_scores$categorical_target %>% factor(levels = c("Positive","Negative"))
confusionMatrix(preds_knn, test$class)
















# how to we choose cp?
set.seed(64)
model_tree_cv <- train(class~.,
                       method = "rpart",
                       tuneGrid = data.frame(cp = seq(0, 0.05, len = 25)),
                       data = diabetes)

ggplot(model_tree_cv, highlight = TRUE)

model_tree_cv$results %>% 
  ggplot(aes(x = cp, y = Accuracy)) +
  geom_line() +
  geom_point() +
  geom_errorbar(aes(x = cp, 
                    ymin = Accuracy - AccuracySD,
                    ymax = Accuracy + AccuracySD))

opt_cp <- model_tree_cv$bestTune

model_tree <- rpart(class~., cp = opt_cp, data = train) 

rpart.plot(model_tree, type = 5)
title("Decision Tree")
rpart.rules(model_tree)

preds_tree <- predict(model_tree, validation) %>%
  as_tibble %$%
  ifelse(Positive > Negative, "Positive", "Negative") %>% 
  factor(levels = c("Positive", "Negative"))
confusionMatrix(preds_tree, validation$class)

















set.seed(11)
model_rf <- train(class~.,
                  method = "rf",
                  tuneGrid = data.frame(mtry = 3:11),
                  cutoff = c(0.3,0.7),
                  data = train)

# ?randomForest for tuning parameters
ggplot(model_rf, highlight = TRUE) +
  scale_x_discrete(limits = 2:12) +
  ggtitle("BLA BLA")
confusionMatrix(predict(model_rf, test),
                test$class)

# consider trying mtry which is the number of random variables to use at each split
importance(rf_1)

model_rf$results %>% 
  ggplot(aes(x = mtry, y = Accuracy)) +
  geom_line() +
  geom_point() +
  geom_errorbar(aes(x = mtry, 
                    ymin = Accuracy - AccuracySD,
                    ymax = Accuracy + AccuracySD))

# look at variable importance with randomforwest
importance(model_rf$finalModel)

