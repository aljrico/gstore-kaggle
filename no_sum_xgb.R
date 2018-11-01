# Libraries & Functions ---------------------------------------------------



library(tidyverse)
library(data.table)
library(jsonlite)
library(lubridate)
library(dataPreparation)
library(moments)

library(tictoc)
tic()
library(caret)
library(xgboost)

source("tuning_xgb.R")
source("feature_engineering.R")

flatten <- function(x){

	pre_flatten <- . %>%
		str_c(., collapse = ",") %>%
		str_c("[", ., "]") %>%
		fromJSON(flatten = T)

	output <- x %>%
		bind_cols(pre_flatten(.$device)) %>%
		bind_cols(pre_flatten(.$geoNetwork)) %>%
		bind_cols(pre_flatten(.$trafficSource)) %>%
		bind_cols(pre_flatten(.$totals)) %>%
		select(-device, -geoNetwork, -trafficSource, -totals)

	return(output)
}

outersect <- function(x, y) {
	sort(c(setdiff(x, y),
				 setdiff(y, x)))
}

numerise_data <- function(data, numeric_columns){
	features <- colnames(data)

	data[is.na(data)] <- 0

	for(f in features) {
		if ((class(data[[f]])=="factor") || (class(data[[f]])=="character")) {
			levels <- unique(data[[f]]) %>% sort()
			data[[f]] <- (factor(data[[f]], levels=levels)) %>% as.numeric()
		}else{
			data[[f]] <- data[[f]] %>% as.numeric()
		}
	}
	data[is.na(data)] <- 0
	return(data)
}

fn <- funs(mean,
					 # sd,
					 # min,
					 # max,
					 # sum,
					 # n_distinct,
					 # kurtosis,
					 # skewness,
					 .args = list(na.rm = TRUE))


# Retrieve Data -----------------------------------------------------------

train <- read_csv("data/train.csv") %>%
	# sample_n(1e4)  %>%
	flatten() %>%
	data.table()

test  <- read_csv("data/test.csv")  %>%  flatten() %>% data.table()



# Define NA ---------------------------------------------------------------

hidden_na <- function(x) x %in% c("not available in demo dataset",
																	"(not provided)",
																	"(not set)",
																	"<NA>",
																	"unknown.unknown",
																	"(none)",
																	"Not Socially Engaged")

train <- train %>%  mutate_all(funs(ifelse(hidden_na(.), NA, .))) %>% data.table()
test  <- test  %>%  mutate_all(funs(ifelse(hidden_na(.), NA, .))) %>% data.table()

train$transactionRevenue[is.na(train$transactionRevenue)] <- 0


# Remove 100% NA variables ------------------------------------------------
train <- train[,which(unlist(lapply(train, function(x)!all(is.na(x))))),with=F]
test <- test[,which(unlist(lapply(test, function(x)!all(is.na(x))))),with=F]




# Classification Data Sets ------------------------------------------------

train_class <- train %>%
	mutate(date = ymd(date)) %>%
	mutate(target_class = (ifelse(transactionRevenue == 0, 0, 1))) %>%
	mutate(target_class = ifelse(target_class > 1, 1, target_class)) %>%
	as_tibble()

test_class <- test %>%
	mutate(date = ymd(date)) %>%
	mutate(target_class = NA) %>%
	as_tibble()

# Classification Feature Engineering -----------------------------------------------------

tri <- 1:nrow(train_class)
y <- train_class$target_class

train_ids <- train_class$fullVisitorId
test_ids <-  test_class$fullVisitorId

tmp <- outersect(colnames(train_class),colnames(test_class))

train_class <- train_class %>% ungroup() %>% as_tibble()

tr_te <- train_class %>%
	select(-campaignCode,-transactionRevenue) %>%
	rbind(test_class) %>%
	data.table()

rm(train_class, test_class, tmp)
gc()

tr_te <- feature_engineering(tr_te) %>% as_tibble()

# Classification Prepare Data -----------------------------------------------------------


tri_val <- sample(tri, length(tri)*0.1)
tr <- tri[!(tri %in% tri_val)]

test_rf <- tr_te %>%
	ungroup() %>%
	as_tibble() %>%
	.[-tri,] %>%
	select(-fullVisitorId,-target_class) %>%
	as_tibble()

train_rf <- tr_te %>%
	ungroup() %>%
	.[tr,] %>%
	rename(target = target_class) %>%
	dplyr::select(-fullVisitorId) %>%
	as_tibble()

true_train <- tr_te %>%
	ungroup() %>%
	.[tri,] %>%
	rename(target = target_class) %>%
	dplyr::select(-fullVisitorId) %>%
	as_tibble()

validation_set <- tr_te %>%
	.[-tri_val,] %>%
	rename(target = target_class) %>%
	dplyr::select(-fullVisitorId) %>%
	as_tibble()

rm(tr_te);gc()


# Classification XGB ------------------------------------------------------

y <- train_rf$target
y_true <- true_train$target
train_xgb <- xgb.DMatrix(data = train_rf %>% select(-target) %>% as.matrix(),
												 label = y)
true_train_xgb <- xgb.DMatrix(data = true_train %>% select(-target) %>% as.matrix(),
												 label = y_true)

val_xgb <- xgb.DMatrix(data = validation_set %>% select(-target) %>% as.matrix(),
											 label = validation_set$target)
test_xgb  <- xgb.DMatrix(data = test_rf %>% as.matrix())

tuning_scores <- tune_xgb(train_data = train_rf,
													target_label = "target",
													ntrees = 100,
													objective = "binary:logistic",
													eval_metric = "auc",
													fast = TRUE)

ts.plot(tuning_scores$scores)

# tuning_scores %>%
# 	melt(id.vars = "scores") %>%
# 	ggplot(aes(y = scores, x = as.factor(value), colour = variable)) +
# 	geom_boxplot() +
# 	facet_grid(.~variable, scales = "free")
#
# tuning_scores %>%
# 	melt(id.vars = "scores") %>%
# 	group_by(variable,value) %>%
# 	summarise(mean_score = mean(scores), sd_score = sd(scores)) %>%
# 	mutate(magic = mean_score - sd_score)

rm(all_ids,test_ids,train_ids);gc()

m <- which.max(tuning_scores$scores)
currentSubsampleRate <- tuning_scores[["subsample"]][[m]]
currentColsampleRate <- tuning_scores[["colsample_bytree"]][[m]]
lr <- tuning_scores[["lr"]][[m]]
mtd <- tuning_scores[["mtd"]][[m]]
mcw <- tuning_scores[["mcw"]][[m]]

ntrees <- 1e3
p <- list(objective = "binary:logistic",
					booster = "gbtree",
					eval_metric = "auc",
					nthread = 4,
					eta = lr/ntrees,
					max_depth = mtd,
					min_child_weight = 30,
					gamma = 0,
					subsample = currentSubsampleRate,
					colsample_bytree = currentColsampleRate,
					colsample_bylevel = 0.632,
					alpha = 0,
					lambda = 0,
					nrounds = ntrees)


cv_m_xgb <- xgb.train(p, train_xgb, 200, list(val = val_xgb), print_every_n = 50, early_stopping_rounds = 300)

m_xgb <- xgb.train(p, true_train_xgb, p$nrounds, list(val = val_xgb), print_every_n = 50, early_stopping_rounds = 300)


# Classification Cross Validation --------------------------------------------------------

prediction_xgb <- predict(cv_m_xgb,val_xgb, type = "prob")
# prediction_xgb_bal <- predict(m_xgb_bal,val_xgb_bal, type = "prob")
#
 confusionMatrix(data = as.factor(as.numeric(ifelse(prediction_xgb > 0.5, 1, 0))), as.factor(validation_set$target))
# confusionMatrix(data = as.factor(as.numeric(ifelse(prediction_xgb_bal > 0.5, 1, 0))), as.factor(validation_set$target))
#
# prediction_xgb <- (prediction_xgb + prediction_xgb_bal)/2
# confusionMatrix(data = as.factor(as.numeric(ifelse(prediction_xgb > 0.5, 1, 0))), as.factor(validation_set$target))


# Classification Prediction --------------------------------------------------------------

prediction_xgb <- predict(m_xgb,test_xgb, type = "prob")
# prediction_xgb_bal <- predict(m_xgb_bal,test_xgb_bal, type = "prob")

# prediction_xgb <- (prediction_xgb + prediction_xgb_bal)/2

prediction_final <- prediction_xgb

prediction_class <- ifelse(prediction_final >= 0.5, 1, 0)

saveRDS(prediction_class, "prediction_class")


# Regression Data Sets ----------------------------------------------------

train_reg <- train %>%
	mutate(date = ymd(date)) %>%
	mutate(target_reg = as.numeric(transactionRevenue)) %>%
	# filter(target_reg != 0) %>%
	as_tibble()

test_reg <- test %>%
	mutate(date = ymd(date)) %>%
	mutate(target_reg = NA) %>%
	as_tibble()

train_reg$binary_prediction <- predict(m_xgb,true_train_xgb, type = "prob")
test_reg$binary_prediction <- prediction_xgb

rm(test_ids,train_ids,validation_ids,m_xgb,test_rf, train_rf, validation_set, test_xgb, true_train, true_train_xgb, test_xgb_bal, train_xgb, train_xgb_bal,all_ids,prediction_class,prediction_final);gc()


# Regression Feature Engineering ------------------------------------------

# Set target
tri <- 1:nrow(train_reg)
y <- train_reg %>%
	filter(target_reg != 0) %>%
	.$target_reg

# Separate Testand Train Ids
train_ids <- train_reg$fullVisitorId
test_ids <-  test_reg$fullVisitorId

# Build Features Data Set
tmp <- outersect(colnames(train_reg),colnames(test_reg))
train_reg <- train_reg %>% ungroup() %>% as_tibble()

# tr_te <- train_reg[ , -which(names(train_reg) %in% tmp)] %>%
# 	rbind(test_reg %>% ungroup()) %>%
# 	data.table()

tr_te <- train_reg %>%
	select(-campaignCode,-transactionRevenue) %>%
	rbind(test_reg) %>%
	data.table()

rm(train_reg, test_reg, tmp); gc()

tr_te <- feature_engineering(tr_te) %>% as_tibble()


# Regression Preparing Data ---------------------------------------

#Preparing Data
tri_val <- sample(tri, length(tri)*0.1)
tr <- tri[!(tri %in% tri_val)]

test_rf <- tr_te %>%
	ungroup() %>%
	as_tibble() %>%
	.[-tri,] %>%
	select(-fullVisitorId,-target_reg) %>%
	as_tibble()

train_rf <- tr_te %>%
	ungroup() %>%
	.[tr,] %>%
	rename(target = target_reg) %>%
	filter(target > 0) %>%
	dplyr::select(-fullVisitorId) %>%
	as_tibble()

validation_set <- tr_te %>%
	.[-tri_val,] %>%
	rename(target = target_reg) %>%
	dplyr::select(-fullVisitorId) %>%
	filter(target != 0) %>%
	as_tibble()


rm(tr_te);gc()

# Regression XGB ----------------------------------------------------------

y <- train_rf$target
train_xgb <- xgb.DMatrix(data = train_rf %>% select(-target) %>% as.matrix(), label = y)
val_xgb <- xgb.DMatrix(data = validation_set %>% select(-target) %>% as.matrix(), label = validation_set$target)
test_xgb  <- xgb.DMatrix(data = test_rf %>% as.matrix())

tuning_scores <- tune_xgb(train_data = train_rf,
													target_label = "target",
													ntrees = 100,
													objective = "reg:linear",
													eval_metric = "rmse",
													fast = TRUE)

ts.plot(tuning_scores$scores)

rm(all_ids,test_ids,train_ids);gc()

m <- which.max(tuning_scores$scores)
currentSubsampleRate <- tuning_scores[["subsample"]][[m]]
currentColsampleRate <- tuning_scores[["colsample_bytree"]][[m]]
lr <- tuning_scores[["lr"]][[m]]
mtd <- tuning_scores[["mtd"]][[m]]

ntrees <- 5e3
p <- list(objective = "reg:linear",
					booster = "gbtree",
					eval_metric = "rmse",
					nthread = 4,
					eta = lr/ntrees,
					max_depth = mtd,
					min_child_weight = 30,
					gamma = 0,
					subsample = currentSubsampleRate,
					colsample_bytree = currentColsampleRate,
					colsample_bylevel = 0.632,
					alpha = 0,
					lambda = 0,
					nrounds = ntrees)

m_xgb <- xgb.train(p, train_xgb, p$nrounds, list(val = val_xgb), print_every_n = 50, early_stopping_rounds = 300)



# Regression Cross Validation ---------------------------------------------
prediction_xgb <- predict(m_xgb, val_xgb) %>% c()

prediction_xgb[prediction_xgb < 0] <- 0

plot(prediction_xgb,validation_set$target)
sum((prediction_xgb-validation_set$target)^2)

xgb.importance(model = m_xgb) %>% as_tibble() %>% top_n(25, Gain) %>%
	ggplot(aes(x = reorder(Feature, Gain), y = Gain)) +
	geom_col() +
	coord_flip()

# Submission --------------------------------------------------------------

id <- test$fullVisitorId %>% as_tibble() %>% set_names("fullVisitorId")
submit <- . %>%
	as_tibble() %>%
	set_names("y") %>%
	mutate(y = ifelse(y < 0, 0, (y))) %>%
	bind_cols(id) %>%
	group_by(fullVisitorId) %>%
	summarise(y = log1p(1 + sum(y))) %>%
	right_join(
		read_csv("data/sample_submission.csv"),
		by = "fullVisitorId") %>%
	mutate(PredictedLogRevenue = round(y, 5)) %>%
	select(-y) %>%
	write_csv("data/submission_new.csv")

prediction_reg <- predict(m_xgb, test_xgb)
prediction_reg[prediction_reg < 0] <- 0
prediction_class <- readRDS("prediction_class")

submit(prediction_reg*prediction_class)

sub <- read_csv("data/sample_submission.csv")
sub$PredictedLogRevenue <- (prediction_class)*prediction_reg %>% c()

sub %>%
	fwrite("data/submission.csv")

toc()
