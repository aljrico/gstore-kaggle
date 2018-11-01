# Libraries  ---------------------------------------------------

library(tidyverse)
library(data.table)
library(magrittr)
library(lubridate)

library(jsonlite)
library(dataPreparation)
library(moments)
library(mlr)
library(stringr)

library(rsample)
library(rlist)

library(tictoc)
tic()
library(caret)
library(xgboost)



# Custom Functions --------------------------------------------------------
source("tuning_xgb.R")
source("feature_engineering.R")
source("munging_functions.R")

# Preprocessing ----------------------------------------------------------------
source("preprocessing.R");gc()

train %<>% data.table()
test  %<>% data.table()

# Define Targets
y <- log1p(as.numeric(train$transactionRevenue))
y[is.na(y)] <- 0
y_class <- ifelse(y>0,1,0)

tri <- 1:nrow(train)
tr_id <- train$fullVisitorId
te_id <- test$fullVisitorId

train[, transactionRevenue := NULL]
train[, campaignCode := NULL]
train %<>% as_tibble()

# Get Folds
tr_val_ind <- get_folds(train, "fullVisitorId", 5)

# Build tr_te
tr_te <- train %>%
	bind_rows(test) %>%
	select_if(has_many_values) %>%
	data.table()

# Basic Feature Engineering
tr_te <- feature_engineering(tr_te) %>% as_tibble()
rm(train,test);gc()

# Training First Iteration -------------------------------------------
dtest <- xgb.DMatrix(data = data.matrix(tr_te[-tri, ]))
pred_tr <- rep(0, length(tri))
pred_te <- 0

train_tune <- tr_te[tri,]
train_tune$target <- y_class
tuning_scores <- tune_xgb(train_data = train_tune,
													target_label = "target",
													ntrees = 100,
													objective = "binary:logistic",
													eval_metric = "auc",
													fast = TRUE)
rm(train_tune);gc()
ts.plot(tuning_scores$scores)

m <- which.min(tuning_scores$scores)
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

for (i in seq_along(tr_val_ind)) {
	cat("Group fold:", i, "\n")

	tr_ind <- tr_val_ind[[i]]$tr
	val_ind <- tr_val_ind[[i]]$val

	dtrain <- xgb.DMatrix(data = data.matrix(tr_te[tr_ind, ]), label = y_class[tr_ind])
	dval <- xgb.DMatrix(data = data.matrix(tr_te[val_ind, ]), label = y_class[val_ind])

	set.seed(0)
	cv <- xgb.train(p, dtrain, p$nrounds, list(val = dval), print_every_n = 50, early_stopping_rounds = 300)

	pred_tr[val_ind] <- expm1(predict(cv, dval))
	pred_te <- pred_te + expm1(predict(cv, dtest))

	rm(dtrain, dval, tr_ind, val_ind)
	gc()
}

pred_tr <- ifelse(pred_tr < 0, 0, pred_tr)
pred_te <- ifelse(pred_te < 0, 0, pred_te / length(tr_val_ind))

cols <- colnames(tr_te)
imp <- xgb.importance(cols, model = cv) %>%
	xgb.plot.importance(top_n = 25)

rm(dtest, cv, train_tune); gc()

saveRDS(pred_tr, "pred_tr")
saveRDS(pred_te, "pred_te")


# MetaFeatures ------------------------------------------

tr_preds <- tr_te %>% dplyr::slice(tri)
tr_preds[, fullVisitorId := tr_id]
tr_preds[, pred := pred_tr]
tr_preds <- tr_preds %>%
	dplyr::select(fullVisitorId,pred) %>%
	group_by(fullVisitorId) %>%
	mutate(pred_num = str_c("pred", 1:n())) %>%
	spread(pred_num,pred) %>%
	data.table()
tr_preds[, p_mean := apply(select(., starts_with("pred")), 1, mean, na.rm = TRUE)]
tr_preds[, p_sd   := apply(select(., starts_with("pred")), 1, sd, na.rm = TRUE)]

te_preds <- tr_te %>% dplyr::slice(-tri)
te_preds[, fullVisitorId := tr_id]
te_preds[, pred := pred_te]
te_preds <- te_preds %>%
	dplyr::select(fullVisitorId,pred) %>%
	group_by(fullVisitorId) %>%
	mutate(pred_num = str_c("pred", 1:n())) %>%
	spread(pred_num,pred) %>%
	data.table()
te_preds[, p_mean := apply(select(., starts_with("pred")), 1, mean, na.rm = TRUE)]
te_preds[, p_sd   := apply(select(., starts_with("pred")), 1, sd, na.rm = TRUE)]

tr_te_preds <- bind_rows(tr_preds, te_preds)
tr_te %<>% cbind(tr_te_preds) %>% data.table()
rm(tr_preds,te_preds);gc()


# Train Second Iteration --------------------------------------------------

train <- tr_te[tri,]
test <- tr_te[-tri,]
rm(tr_te);gc()

dtest <- xgb.DMatrix(data = data.matrix(test))

pred_tr <- rep(0, nrow(train))
pred_te <- 0
err <- 0

train$target <- y
tuning_scores <- tune_xgb(train_data = train,
													target_label = "target",
													ntrees = 100,
													objective = "reg:linear",
													eval_metric = "rmse",
													fast = TRUE)
rm(train_tune);gc()
train$target <- NULL
ts.plot(tuning_scores$scores)

m <- which.min(tuning_scores$scores)
currentSubsampleRate <- tuning_scores[["subsample"]][[m]]
currentColsampleRate <- tuning_scores[["colsample_bytree"]][[m]]
lr <- tuning_scores[["lr"]][[m]]
mtd <- tuning_scores[["mtd"]][[m]]
mcw <- tuning_scores[["mcw"]][[m]]

for (i in seq_along(tr_val_ind)) {
	cat("Group fold:", i, "\n")

	tr_ind <- tr_val_ind[[i]]$tr
	val_ind <- tr_val_ind[[i]]$val

	dtrain <- xgb.DMatrix(data = data.matrix(train[tr_ind, ]), label = y[tr_ind])
	dval <- xgb.DMatrix(data = data.matrix(train[val_ind, ]), label = y[val_ind])

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

	set.seed(0)
	cv <- xgb.train(p, dtrain, 5000, list(val = dval),
									print_every_n = 150, early_stopping_rounds = 250)

	pred_tr[val_ind] <- predict(cv, dval)
	pred_te <- pred_te + predict(cv, dtest)
	err <- err + cv$best_score

	rm(dtrain, dval, tr_ind, val_ind, p)
	gc()
}

pred_tr <- ifelse(pred_tr < 0, 0, pred_tr)
pred_te <- ifelse(pred_te < 0, 0, pred_te / length(tr_val_ind))
err <- err / length(tr_val_ind)

# Submit ------------------------------------------------------------------

submit <- . %>%
	as_tibble() %>%
	set_names("y") %>%
	mutate(y = ifelse(y < 0, 0, expm1(y))) %>%
	bind_cols(id) %>%
	group_by(fullVisitorId) %>%
	summarise(y = log1p(sum(y))) %>%
	right_join(
		read_csv("data/sample_submission.csv"),
		by = "fullVisitorId") %>%
	mutate(PredictedLogRevenue = round(y, 5)) %>%
	select(-y) %>%
	write_csv("data/submission_new.csv")

id <- test$fullVisitorId %>% as_tibble() %>% set_names("fullVisitorId")
submit(pred_te)
