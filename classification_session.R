classification_session <- function(train,test, fast = TRUE){
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
		select(-transactionRevenue, -campaignCode) %>%
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
														fast = fast)

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


	cv_model_class_sess_xgb <- xgb.train(p, train_xgb, 200, list(val = val_xgb), print_every_n = 50, early_stopping_rounds = 300)

	model_class_sess_xgb <<- xgb.train(p, true_train_xgb, p$nrounds, list(val = val_xgb), print_every_n = 50, early_stopping_rounds = 300)


	# Classification Cross Validation --------------------------------------------------------

	prediction_xgb <- predict(cv_model_class_sess_xgb,val_xgb, type = "prob")
	confusionMatrix(data = as.factor(as.numeric(ifelse(prediction_xgb > 0.5, 1, 0))), as.factor(validation_set$target))
	# Classification Prediction --------------------------------------------------------------

	prediction_xgb <- predict(model_class_sess_xgb,test_xgb, type = "prob")
	# prediction_xgb_bal <- predict(model_class_sess_xgb_bal,test_xgb_bal, type = "prob")

	# prediction_xgb <- (prediction_xgb + prediction_xgb_bal)/2

	prediction_final <- prediction_xgb

	prediction_class <- ifelse(prediction_final >= 0.5, 1, 0)

	saveRDS(prediction_class, "prediction_class")
	return(prediction_class)
}
