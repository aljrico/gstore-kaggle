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
library(progress)

library(caret)
library(xgboost)


tic()

# --------------------
cat("Custom Functions \n ")

tune_xgb <- function(train_data, target_label, ntrees = 100, objective = "binary:logistic", eval_metric = "error", fast = TRUE){
	train_data <- as.data.frame(train_data)

	# Count Event Rate
	if(objective == "binary:logistic") event_rate <- ceiling(1/(sum(train_data$target == 1)/length(train_data$target)))
	if(!(objective == "binary:logistic")) event_rate <-  10
	if(fast){
		parameterList <- expand.grid(subsample = seq(from = 1, to = 1, by = 1),
																 colsample_bytree = seq(from = 0.5, to = 1, by = 0.5),
																 lr = seq(from = 2, to = 10, by = 2),
																 mtd = seq(from = 4, to = 10, by = 2),
																 mcw = seq(from = event_rate, to = event_rate, by = event_rate))
	}else{
		parameterList <- expand.grid(subsample = seq(from = 0.5, to = 1, by = 0.5),
																 colsample_bytree = seq(from = 0.4, to = 1, by = 0.2),
																 lr = seq(from = 1, to = 15, by = 1),
																 mtd = seq(from = 2, to = 16, by = 2),
																 mcw = seq(from = floor(event_rate/2), to = event_rate*10, by = floor(event_rate*2)))
	}
	scores <- c()

	pb <- progress_bar$new(
		format = " Tuning Hyperparameters [:bar] :percent eta: :eta",
		total = nrow(parameterList), clear = FALSE, width= 60)

	for(i in 1:nrow(parameterList)){
		pb$tick()
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

		if(eval_metric == "auc") scores[i] <- xgb_cv$evaluation_log$test_auc_mean %>% max()
		if(eval_metric == "error") scores[i] <- xgb_cv$evaluation_log$test_error_mean %>% min()
		if(eval_metric == "rmse") scores[i] <- xgb_cv$evaluation_log$test_rmse_mean %>% min()
	}
	parameterList$scores <- scores
	return(parameterList)
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


get_folds <- function(data, group, v = 5) {
	group_folds <- group_vfold_cv(data[group], group, v = v)
	list.zip(tr = tr_ind <- map(group_folds$splits, ~ .x$in_id),
					 val = val_ind <- map(group_folds$splits, ~ setdiff(1:nrow(data), .x$in_id)))
}

create_time_fea <- function(fun = lag, n = 1){
	tr_tmp <- tr_te %>% dplyr::slice(tri) %>% data.table()
	te_tmp <- tr_te %>% dplyr::slice(-tri) %>% data.table()

	tr_tmp[, date := as_datetime(visitStartTime)]
	te_tmp[, date := as_datetime(visitStartTime)]

	tr_tmp[, time_var := (date - fun(date, n)) %>% as.integer()/3600]
	te_tmp[, time_var := (date - fun(date, n)) %>% as.integer()/3600]

	output <- c(tr_tmp$time_var, te_tmp$time_var)

	rm(tr_tmp, te_tmp)
	return(output)
}
has_many_values <- function(x) n_distinct(x) > 1


# -----------------
cat("Preprocessing \n")

# Retrieve Data
train <- read_csv("data/train.csv") %>% sample_n(1e4) %>% flatten() %>% data.table()
test  <- read_csv("data/test.csv")  %>% flatten() %>% data.table()


# Define and remove NA
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


# Remove 100% NA variables
train <- train[,which(unlist(lapply(train, function(x)!all(is.na(x))))),with=F]
test <- test[,which(unlist(lapply(test, function(x)!all(is.na(x))))),with=F]


# Formatting
train[, date := ymd(date)]
test[,  date := ymd(date)]


# Remove Outliers
train %>% filter(country != 'Anguilla') %>% data.table()

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
tr_val_ind <- get_folds(train, "fullVisitorId", 2)

# Build tr_te
tr_te <- train %>%
	bind_rows(test) %>%
	select_if(has_many_values) %>%
	data.table()

# --------------------------------------
cat("Feature Engineering: \n")
feature_engineering <- function(tr_te, models = NULL, tri = 0){
	tr_te <- data.table(tr_te)
	# Building Simple Features
	cat("... Building Simple Features ... \n")
	tr_te[, weekend := ifelse((date %>% wday(label = TRUE, abbr = FALSE)) %in% c("Saturday", "Sunday"),1,0)]
	tr_te[, wday := date %>% wday(label = TRUE, abbr = FALSE)]
	tr_te[, year := date %>% year()]
	tr_te[, month := date %>% month()]
	tr_te[, hour := as.numeric(hour(as_datetime(visitStartTime)))]
	tr_te[, browser := ifelse(browser %in% c("Safari", "Firefox"), "mainstream", ifelse(browser == "Chrome", 'Chrome', "Other"))]
	tr_te[, is_chrome_the_browser := ifelse(browser == "Chrome", 1, 0) %>% as.numeric()]
	tr_te[, source_from_googleplex := ifelse(source == 'mail.googleplex.com', 1, 0) %>% as.numeric()]
	tr_te[, source_from_youtube := ifelse(source == 'youtube.com', 1, 0) %>% as.numeric()]
	tr_te[, is_medium_referral := ifelse(medium == 'referral', 1, 0) %>% as.numeric()]
	tr_te[, is_device_desktop := ifelse(deviceCategory == 'desktop', 1, 0) %>% as.numeric()]
	tr_te[, is_device_macbook := is_device_desktop*ifelse(operatingSystem == "Macintosh", 1, 0)]
	tr_te[, windows_desktop := is_device_desktop*ifelse(operatingSystem == 'Windows', 1, 0)]
	tr_te[, is_device_chromebook := ifelse(operatingSystem == "Chrome OS", 1, 0)]
	tr_te[, is_device_linux := ifelse(operatingSystem == "Linux", 1, 0)]
	tr_te[, is_phone_ios := ifelse(operatingSystem == "iOS", 1, 0)]
	tr_te[, is_phone_android := ifelse(operatingSystem == "Android", 1, 0)]
	tr_te[, is_phone_windows := ifelse(operatingSystem == "Windows Phone", 1, 0)]
	tr_te[, single_visit := ifelse(visitNumber == 1,1,0) ]
	tr_te[, hits_ratio := as.numeric(hits)/as.numeric(pageviews)]
	tr_te[, domain_site := gsub("^.*\\.","", networkDomain)]
	tr_te[, adwordsClickInfo.isVideoAd := ifelse(is.na(adwordsClickInfo.isVideoAd), 1, 0)]
	tr_te[, source_shared := ifelse(grepl("shared", source), 1, 0)]
	tr_te[, source_platform := ifelse(grepl("login", source), 1, 0)]
	tr_te[, source_depth := str_count(source, '.')]
	tr_te[, source_length := nchar(source)]
	tr_te[, source_complexity := source_depth/source_length]
	tr_te[, path_depth := str_count(referralPath, '/')]
	tr_te[, path_length := nchar(referralPath)]
	tr_te[, path_complexity := path_depth/path_length]
	# tr_te[,which(unlist(lapply(tr_te, function(x)!all(is.na(x))))),with=F]

	# Manual Combinations
	cat("... Manual Combinations ... \n")
	tr_te[, browser_dev := str_c(browser, "_", deviceCategory)]
	tr_te[, browser_os := str_c(browser, "_", operatingSystem)]
	tr_te[, browser_chan := str_c(browser,  "_", channelGrouping)]
	tr_te[, campaign_medium := str_c(campaign, "_", medium)]
	tr_te[, chan_os := str_c(operatingSystem, "_", channelGrouping)]
	tr_te[, country_adcontent := str_c(country, "_", adContent)]
	tr_te[, country_medium := str_c(country, "_", medium)]
	tr_te[, country_source := str_c(country, "_", source_depth)]
	tr_te[, dev_chan := str_c(deviceCategory, "_", channelGrouping)]

	tr_te %<>% as_tibble()

	# Dummy Variables
	cat("... Dummy Variables ... \n")
	small_features <- c('channelGrouping','deviceCategory','adwordsClickInfo.slot','adwordsClickInfo.adNetworkType','medium','continent')
	dummies <- caret::dummyVars( ~ ., data = tr_te[small_features], fullRank=T)
	tr_te %<>% cbind(predict(dummies, tr_te))

	# Automatic Combinations
	cat("... Automatic Combinations ... \n")
	for (i in c("city", "continent", "country", "metro", "domain_site", "region", "subContinent"))
		for (j in c("browser", "deviceCategory", "operatingSystem", "source_depth"))
			tr_te[str_c(i, "_", j)] <- str_c(tr_te[[i]], tr_te[[j]], sep = "_")

	# Time Features
	cat("... Time Features ... \n")
	for (i in 1:5) {
		tr_te[str_c("next_sess", i)] <- create_time_fea(lag, i)
		tr_te[str_c("prev_sess", i)] <- create_time_fea(lead, i)
	}

	# Time Tendency
	cat("... Time Tendency ... \n")
	# tr_te %<>%
	# 	dplyr::mutate(tendency_prev = lm(c(prev_sess1,prev_sess2,prev_sess3,prev_sess4,prev_sess5)~c(1,2,3,4,5))%>% .$coefficients %>% .[[2]]) %>%
	# 	dplyr::mutate(tendency_next = lm(c(next_sess1,next_sess2,next_sess3,next_sess4,next_sess5)~c(1,2,3,4,5))%>% .$coefficients %>% .[[2]])
	#
	# tmp_dep <- tr_te[,c("prev_sess1","prev_sess2","prev_sess2", "prev_sess3", "prev_sess4", "prev_sess5")]

	ps1 <- ifelse(is.na(tr_te$prev_sess1), 10, tr_te$prev_sess1)
	ps2 <- ifelse(is.na(tr_te$prev_sess1), 10, tr_te$prev_sess2)
	ps3 <- ifelse(is.na(tr_te$prev_sess1), 10, tr_te$prev_sess3)
	ps4 <- ifelse(is.na(tr_te$prev_sess1), 10, tr_te$prev_sess4)
	ps5 <- ifelse(is.na(tr_te$prev_sess1), 10, tr_te$prev_sess5)
	tendency_prev <- c()
	for(i in 1:length(ps1)) tendency_prev[[i]] <- lm(c(ps1[[i]],ps2[[i]],ps3[[i]],ps4[[i]],ps5[[i]])~c(1,2,3,4,5)) %>% .$coefficients %>% .[[2]]
	tr_te$tendency_prev <- tendency_prev
	rm(ps1,ps2,ps3,ps4,ps5);gc()

	tr_te %<>% data.table()
	all_ids <- tr_te$fullVisitorId

	tr_te <- tr_te %>%
		select(-visitId, -sessionId, -fullVisitorId) %>%
		numerise_data() %>%
		as_tibble()

	tr_te %<>% data.table()


	# Clean
	cat("... Cleaning up ... \n")
	tr_te[, date := NULL]
	tr_te[, visitStartTime := NULL]
	tr_te[, source := NULL]
	tr_te[, medium := NULL]
	tr_te[, browser := NULL]
	tr_te[, operatingSystem := NULL]
	tr_te[, deviceCategory := NULL]
	tr_te[, visitNumber := NULL]
	tr_te[, networkDomain := NULL]

	# Remove redundant columns
	tr_te <- tr_te[,which(unlist(lapply(tr_te, function(x)!all(is.na(x))))),with=F]
	constant_columns <- whichAreConstant(tr_te, verbose=FALSE)
	if(length(constant_columns > 0)) tr_te <- subset(tr_te,select = -c(constant_columns)) %>% as_tibble()

	tr_te[is.na(tr_te)] <- 0
	return(tr_te)
}

# Basic Feature Engineering
tr_te <- feature_engineering(tr_te) %>% as_tibble()
rm(train,test);gc()


cat("Training First Iteration: \n")
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

for (i in seq_along(tr_val_ind)) {
	cat("Group fold:", i, "\n")

	tr_ind <- tr_val_ind[[i]]$tr
	val_ind <- tr_val_ind[[i]]$val

	dtrain <- xgb.DMatrix(data = data.matrix(tr_te[tr_ind, ]), label = y_class[tr_ind])
	dval <- xgb.DMatrix(data = data.matrix(tr_te[val_ind, ]), label = y_class[val_ind])

	cv <- xgb.train(p, dtrain, p$nrounds, list(val = dval), print_every_n = 50, early_stopping_rounds = 300)

	pred_tr[val_ind] <- (predict(cv, dval, type = "prob"))
	pred_te <- pred_te + (predict(cv, dtest, type = "prob"))

	rm(dtrain, dval, tr_ind, val_ind)
	gc()
}

pred_tr <- ifelse(pred_tr < 0, 0, pred_tr)
pred_te <- ifelse(pred_te < 0, 0, pred_te / length(tr_val_ind))
pred_class <- pred_te

xgb.importance(model = cv) %>% as_tibble() %>% top_n(25, Gain) %>%
	ggplot(aes(x = reorder(Feature, Gain), y = Gain)) +
	geom_col() +
	coord_flip()

rm(dtest, cv); gc()

# MetaFeatures ------------------------------------------
cat(" Add Meta-Features \n")

class_pred <- c(pred_tr,pred_te)
class_pred_te <- pred_te
tr_te$meta_feature <- class_pred


# Train Second Iteration --------------------------------------------------
cat("Training Second Iteration")

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

	cv <- xgb.train(p, dtrain, p$nrounds, list(val = dval), print_every_n = 50, early_stopping_rounds = 300)


	pred_tr[val_ind] <- predict(cv, dval)
	pred_te <- pred_te + predict(cv, dtest)
	err <- err + cv$best_score

	rm(dtrain, dval, tr_ind, val_ind, p)
	gc()
}

xgb.importance(model = cv) %>% as_tibble() %>% top_n(25, Gain) %>%
	ggplot(aes(x = reorder(Feature, Gain), y = Gain)) +
	geom_col() +
	coord_flip()

pred_tr <- ifelse(pred_tr < 0, 0, pred_tr)
pred_te <- ifelse(pred_te < 0, 0, pred_te / length(tr_val_ind))
err <- err / length(tr_val_ind)


# Munging data at user level ---------------------------
cat("Munging data at user level:\n")

y_ul <- data.table(fullVisitorId = tr_id, y = expm1(y))
y_ul[, y := sum(y), by = fullVisitorId]


cat(" ... Summarising training data ... \n")
tr_ul <- train %>% mutate(fullVisitorId = tr_id) %>% data.table()
tr_ul <- tr_ul[, lapply(.SD, mean, na.rm=TRUE), by=fullVisitorId ]

tr_preds <- train %>%
	mutate(fullVisitorId = tr_id,
				 pred = pred_tr) %>%
	select(fullVisitorId, pred) %>%
	group_by(fullVisitorId) %>%
	mutate(pred_num = str_c("pred", 1:n())) %>%
	ungroup() %>%
	spread(pred_num, pred) %>%
	mutate(      p_mean = apply(select(., starts_with("pred")), 1, mean, na.rm = TRUE),
							 p_med  = apply(select(., starts_with("pred")), 1, median, na.rm = TRUE),
							 p_sd   = apply(select(., starts_with("pred")), 1, sd, na.rm = TRUE),
							 p_sum  = apply(select(., starts_with("pred")), 1, sum, na.rm = TRUE),
							 p_min  = apply(select(., starts_with("pred")), 1, min, na.rm = TRUE),
							 p_max  = apply(select(., starts_with("pred")), 1, max, na.rm = TRUE)) %>% data.table()

# join_columns <- intersect(outersect(colnames(tr_preds), colnames(tr_ul)), colnames(tr_ul))
#
# tr_ul %>% left_join(tr_preds, on = "fullVisitorId") %>% head()
#
# tr_ul[tr_preds, on = "fullVisitorId", mget(paste0("i.", join_columns)) := paste0(join_columns)]
#
# join_columns <- outersect(colnames(y_ul), colnames(tr_ul)) %>% intersect(colnames(tr_ul))
# tr_ul[y_ul, on = "fullVisitorId", paste0(join_columns) := mget(paste0("i.", join_columns))]
#

tr_ul %<>%
	left_join(tr_preds, by = "fullVisitorId") %>%
	left_join(y_ul, by = "fullVisitorId")

y_ul <- tr_ul$y
tr_ul$y <- NULL

rm(tr_preds)
gc()

cat(" ... Summarising test data \n")
te_ul <- test %>% mutate(fullVisitorId = te_id) %>% data.table()
te_ul[, lapply(.SD, mean, na.rm=TRUE), by=fullVisitorId ]

te_preds <- test %>%
	mutate(fullVisitorId = te_id,
				 pred = pred_te) %>%
	select(fullVisitorId, pred) %>%
	group_by(fullVisitorId) %>%
	mutate(pred_num = str_c("pred", 1:n())) %>%
	ungroup() %>%
	spread(pred_num, pred) %>%
	mutate(p_mean = apply(select(., starts_with("pred")), 1, mean, na.rm = TRUE),
				 p_med  = apply(select(., starts_with("pred")), 1, median, na.rm = TRUE),
				 p_sd   = apply(select(., starts_with("pred")), 1, sd, na.rm = TRUE),
				 p_sum  = apply(select(., starts_with("pred")), 1, sum, na.rm = TRUE),
				 p_min  = apply(select(., starts_with("pred")), 1, min, na.rm = TRUE),
				 p_max  = apply(select(., starts_with("pred")), 1, max, na.rm = TRUE)) %>% data.table()

# join_columns <- outersect(colnames(te_preds), colnames(te_ul)) %>% intersect(colnames(te_preds))
# te_ul[te_preds, on = "fullVisitorId", paste0(join_columns) := mget(paste0("i.", join_columns))]

te_ul %<>%
	left_join(te_preds, by = "fullVisitorId")

te_id <- te_ul$fullVisitorId
te_ul$fullVisitorId <- NULL

rm(te_preds)
gc()

tr_val_ind <- get_folds(tr_ul, "fullVisitorId", 2)
tr_ul$fullVisitorId <- NULL

cols <- intersect(names(tr_ul), names(te_ul))
tr_ul %<>% dplyr::select(cols)
te_ul %<>% dplyr::select(cols)

rm(cols, tr_id, tri, y)
gc()

# Training Third Iteration (user level) ---------------------------
cat("Training Third Iteration (user level) \n")

train <- tr_ul
test <- te_ul
rm(tr_ul,te_ul);gc()

dtest <- xgb.DMatrix(data = data.matrix(test))

pred_tr <- rep(0, nrow(train))
pred_te <- 0
err <- 0

y_ul %<>% log1p()
train$target <- y_ul
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

	dtrain <- xgb.DMatrix(data = data.matrix(train[tr_ind, ]), label = y_ul[tr_ind])
	dval <- xgb.DMatrix(data = data.matrix(train[val_ind, ]), label = y_ul[val_ind])

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

	cv <- xgb.train(p, dtrain, p$nrounds, list(val = dval), print_every_n = 50, early_stopping_rounds = 300)


	pred_tr[val_ind] <- predict(cv, dval)
	pred_te <- pred_te + predict(cv, dtest)
	err <- err + cv$best_score

	rm(dtrain, dval, tr_ind, val_ind, p)
	gc()
}

pred_tr <- ifelse(pred_tr < 0, 0, pred_tr)
pred_te <- ifelse(pred_te < 0, 0, pred_te / length(tr_val_ind))
err <- err / length(tr_val_ind)


xgb.importance(model = cv) %>% as_tibble() %>% top_n(25, Gain) %>%
	ggplot(aes(x = reorder(Feature, Gain), y = Gain)) +
	geom_col() +
	coord_flip()



# Submit ------------------------------------------------------------------
cat("Submitting \n")
submit <- . %>%
	as_tibble() %>%
	set_names("y") %>%
	mutate(y = ifelse(y < 0, 0, expm1(y))) %>%
	cbind(fullVisitorId) %>%
	group_by(fullVisitorId) %>%
	summarise(y = log1p(sum(y))) %>%
	right_join(
		read_csv("data/sample_submission.csv"),
		by = "fullVisitorId") %>%
	mutate(PredictedLogRevenue = round(y, 5)) %>%
	select(-y) %>%
	write_csv(paste0("double_iteration_xgb_", round(err, 5), ".csv"))

fullVisitorId <- te_id
submit(pred_te)

toc()


