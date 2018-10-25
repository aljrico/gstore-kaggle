tune_xgb <- function(train_data, target_label, ntrees = 100, objective = "binary:logistic", eval_metric = "error"){


	train_data <- as.data.frame(train_data)

	# Count Event Rate
	if(objective == "binary:logistic") event_rate <- ceiling(1/(sum(train_data$target == 1)/length(train_data$target)))
	if(!(objective == "binary:logistic")) event_rate <-  10

	parameterList <- expand.grid(subsample = seq(from = 0.5, to = 1, by = 0.5),
															 colsample_bytree = seq(from = 0.4, to = 1, by = 0.2),
															 lr = seq(from = 2, to = 10, by = 1),
															 mtd = seq(from = 4, to = 10, by = 2),
															 mcw = seq(from = event_rate, to = event_rate*5, by = floor(event_rate) ))

															 scores <- c()

															 for(i in 1:nrow(parameterList)){

															 	# Define Subsample of Training Data
															 	sample_size <- floor(nrow(train_data)/100)
															 	sample_size <- max(c(sample_size,1e4))
															 	if(nrow(train_data) <= 1e4) sample_size <- nrow(train_data)
															 	train_params <- train_data %>% sample_n(sample_size)
															 	y_params <- train_params[[target_label]]
															 	train_xgb_params <- xgb.DMatrix(data = train_params[,-which(names(train_params) %in% target_label)] %>% as.matrix(),
															 																	label = y_params)

															 	#Extract Parameters to test
															 	currentSubSample <- parameterList[["subsample"]][[i]]
															 	currentColsampleRate <- parameterList[["colsample_bytree"]][[i]]
															 	lr <- parameterList[["lr"]][[i]]
															 	mtd <- parameterList[["mtd"]][[i]]
															 	mcw <- parameterList[["mcw"]][[i]]

															 	p <- list(objective = objective,
															 						booster = "gbtree",
															 						eval_metric = eval_metric,
															 						nthread = 4,
															 						eta = lr/ntrees,
															 						max_depth = mtd,
															 						min_child_weight = mcw,
															 						gamma = 0,
															 						subsample = currentSubSample,
															 						colsample_bytree = currentColsampleRate,
															 						colsample_bylevel = 0.632,
															 						alpha = 0,
															 						lambda = 0,
															 						nrounds = ntrees)

															 	xgb_cv <- xgb.cv(p, train_xgb_params, p$nrounds, print_every_n = 5, early_stopping_rounds = 25, nfold = 5, verbose = 0)
															 	cat(paste0("... ", floor(i/nrow(parameterList)*100), " (%)  ... \n"))
															 	if(eval_metric == "auc") scores[i] <- xgb_cv$evaluation_log$test_auc_mean %>% max()

															 	if(eval_metric == "error") scores[i] <- xgb_cv$evaluation_log$test_error_mean %>% min()


															 	if(eval_metric == "rmse") scores[i] <- xgb_cv$evaluation_log$test_rmse_mean %>% min()
															 }
															 parameterList$scores <- scores
															 return(parameterList)
}
