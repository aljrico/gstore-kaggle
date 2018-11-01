classification_user <- function(train,test, fast = TRUE, models_to_feature){

	# Classification Feature Engineering -----------------------------------------------------

	tri <- 1:nrow(train)
	y <- train %>%
		group_by(fullVisitorId) %>%
		summarise(target = sum(as.numeric(transactionRevenue))) %>%
		mutate(target = ifelse(target > 0, 1, 0)) %>%
		.$target

	tmp <- outersect(colnames(train),colnames(test))

	train <- train %>% ungroup() %>% as_tibble()

	tr_te <- train[ , -which(names(train) %in% tmp)] %>%
		rbind(test %>% ungroup()) %>%
		data.table()

	rm(train, test, tmp)
	gc()

	tr_te <- feature_engineering(tr_te) %>% as_tibble()

	# Classification Prepare Data -----------------------------------------------------------

	train <- tr_te[tri,] %>% data.table()
	test <- tr_te[-tri,] %>% data.table()
	train <- train[, lapply(.SD, mean, na.rm=TRUE), by=fullVisitorId ]
	test <-  test[,  lapply(.SD, mean, na.rm=TRUE), by=fullVisitorId ]

	train[, fullVisitorId := NULL]
	test[, fullVisitorId := NULL]

	tri <- 1:nrow(train)
	tri_val <- sample(tri, length(tri)*0.1)
	tr <- tri[!(tri %in% tri_val)]

	val <- train[tri_val,]
	train <- train[-tri_val,]

	rm(tr_te);gc()

	# Classification XGB ------------------------------------------------------

	train$target <- y[-tri_val]
	val$target <- y[tri_val]
	train_xgb <- xgb.DMatrix(data = train %>% select(-target) %>% as.matrix(),
													 label = y[-tri_val])
	val_xgb <- xgb.DMatrix(data = val %>% select(-target) %>%  as.matrix(),
													 label = y[tri_val])
	test_xgb  <- xgb.DMatrix(data = test %>% as.matrix())

	tuning_scores <- tune_xgb(train_data = train,
														target_label = "target",
														ntrees = 100,
														objective = "binary:logistic",
														eval_metric = "auc",
														fast = fast)
	ts.plot(tuning_scores$scores)

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


	model_class_user_xgb <<- xgb.train(p, train_xgb, p$nrounds, list(val = val_xgb), print_every_n = 50, early_stopping_rounds = 300)


	# Prediction
	prediction_xgb <- predict(model_class_user_xgb,test_xgb, type = "prob")
	prediction_class <- ifelse(prediction_xgb >= 0.5, 1, 0) %>% as.numeric()
	return(prediction_class)
}
