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


# Preprocessing ----------------------------------------------------------------
tuning_fast <- TRUE
source("preprocessing.R");gc()

# Define Targets -----------------------------------------------------
y_reg_red <- train %>%
	filter(transactionRevenue > 0) %>%
	mutate(target = log1p(as.numeric(transactionRevenue))) %>%
	.$target

y_reg <- train %>%
	mutate(target = log1p(as.numeric(transactionRevenue))) %>%
	.$target

y_class <- train %>%
	mutate(target = (ifelse(transactionRevenue == 0, 0, 1))) %>%
	mutate(target = ifelse(target > 1, 1, target)) %>%
	.$target


# Feature Engineering -----------------------------------------------------

tri <- 1:nrow(train)
tmp <- outersect(colnames(train),colnames(test))

train <- train %>% ungroup() %>% as_tibble()
test <- test %>% ungroup() %>% as_tibble()

tr_te <- train[ , -which(names(train) %in% tmp)] %>%
	rbind(test) %>%
	data.table()

rm(train, test, tmp)
gc()

tr_te <- feature_engineering(tr_te) %>% as_tibble()


# Build Meta Features ------------------------------------------------------------

# Prepare Data
train <- tr_te[tri,] %>% data.table()
test <- tr_te[-tri,] %>% data.table()

xgb_model <- function(train,test,target, objective_reg = TRUE, fast = TRUE, reduced = FALSE, user_level = FALSE){

	if(objective_reg){
		objective = "reg:linear"
		eval_metric = "rmse"
	}else{
		objective = "binary:logistic"
		eval_metric = "auc"
		target[target > 0] <- 1
	}

	if((reduced & objective_reg)){
		train$target <- target
		train <- train %>% filter(target > 0) %>% select(-target)
		target <- target[target > 0]
	}

	if(user_level){
		train <- train[, lapply(.SD, mean, na.rm=TRUE), by=fullVisitorId ]
		test <-  test[,  lapply(.SD, mean, na.rm=TRUE), by=fullVisitorId ]
	}

	train[, fullVisitorId := NULL]
	test[, fullVisitorId := NULL]

	tri <- 1:nrow(train)
	tri_val <- sample(tri, length(tri)*0.1)

	val <- train[tri_val,]
	train <- train[-tri_val,]

	# Build XGB datasets
	train_xgb <- xgb.DMatrix(data = train %>% as.matrix(), label = target[-tri_val])
	val_xgb <-   xgb.DMatrix(data = val  %>%  as.matrix(), label = target[tri_val])
	test_xgb  <- 		 xgb.DMatrix(data = test %>% as.matrix())

	train_tune <- train
	train_tune$target <- target[-tri_val]
	tuning_scores <- tune_xgb(train_data = train_tune,
														target_label = "target",
														ntrees = 100,
														objective = objective,
														eval_metric = eval_metric,
														fast = fast)

	m <- which.min(tuning_scores$scores)
	currentSubsampleRate <- tuning_scores[["subsample"]][[m]]
	currentColsampleRate <- tuning_scores[["colsample_bytree"]][[m]]
	lr <- tuning_scores[["lr"]][[m]]
	mtd <- tuning_scores[["mtd"]][[m]]
	mcw <- tuning_scores[["mcw"]][[m]]

	ntrees <- 1e3
	p <- list(objective = objective,
						booster = "gbtree",
						eval_metric = eval_metric,
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


	model <- xgb.train(p, train_xgb, p$nrounds, list(val = val_xgb), print_every_n = 50, early_stopping_rounds = 300)
	# prediction <- predict(model,test_xgb)
	return(model)
}

meta_features <- list()
for(i in 1:3){
	train_sample <- train
	train_sample$target <- y_reg
	train_sample <- train_sample %>% sample_n(5e4)
	y_sample <- train_sample$target
	train_sample <- train_sample %>% select(-target)
	meta_features[[i]] <- xgb_model(train = train_sample, test, target = y_sample, objective_reg = TRUE, fast = TRUE, reduced = FALSE, user_level = FALSE)
}

# for(j in 6:10){
# 	train_sample <- train
# 	train_sample$target <- y_reg
# 	train_sample <- train_sample %>% sample_n(5e4)
# 	y_sample <- train_sample$target
# 	train_sample <- train_sample %>% select(-target)
# 	meta_features[[j]] <- xgb_model(train = train_sample, test, target = y_sample, objective_reg = FALSE, fast = TRUE, reduced = FALSE, user_level = FALSE)
# }


tr_te <- data.table(tr_te)
tr_te[, fullVisitorId := NULL]
tr_te <- as_tibble(tr_te)
models <- meta_features
predictions <- list()
for(i in 1:length(models)){
	m <- models[[i]]
	predictions[[i]] <- predict(m, xgb.DMatrix(data = tr_te %>% as.matrix()))
	cat(paste0(i, "... \n"))
}
for(i in 1:length(models)){
	m <- models[[i]]
	name <- paste0("meta_feature_", i)
	tr_te[name] <- predictions[[i]]
}
