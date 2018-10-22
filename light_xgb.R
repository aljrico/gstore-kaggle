# Libraries & Functions ---------------------------------------------------



library(tidyverse)
library(data.table)
library(jsonlite)
library(lubridate)
library(moments)

library(tictoc)
tic()
library(caret)
library(xgboost)

flatten <- function(x){

	pre_flatten <- . %>%
		str_c(., collapse = ",") %>%
		str_c("[", ., "]") %>%
		fromJSON(flatten = T)

	x %>%
		bind_cols(pre_flatten(.$device)) %>%
		bind_cols(pre_flatten(.$geoNetwork)) %>%
		bind_cols(pre_flatten(.$trafficSource)) %>%
		bind_cols(pre_flatten(.$totals)) %>%
		select(-device, -geoNetwork, -trafficSource, -totals)	%>%
		return()
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
					 sd,
					 min,
					 max,
					 sum,
					 n_distinct,
					 kurtosis,
					 skewness,
					 .args = list(na.rm = TRUE))


# Retrieve Data -----------------------------------------------------------

train <- read_csv("data/train.csv") %>% sample_n(1e4)  %>% flatten() %>% data.table()
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
train[,which(unlist(lapply(train, function(x)!all(is.na(x))))),with=F]
test[,which(unlist(lapply(test, function(x)!all(is.na(x))))),with=F]




# Classification Data Sets ------------------------------------------------

train_class <- train %>%
	mutate(date = ymd(date)) %>%
	group_by(fullVisitorId, date) %>%
	mutate(target_class = sum(ifelse(transactionRevenue == 0, 0, 1))) %>%
	mutate(target_class = ifelse(target_class > 1, 1, target_class)) %>%
	as_tibble()

median_revenue <- train %>%
	group_by(fullVisitorId) %>%
	mutate(target_median = sum(transactionRevenue %>% as.numeric() %>% log())) %>%
	filter(target_median > 0) %>%
	.$target_median %>%
	median()

test_class <- test %>%
	mutate(date = ymd(date)) %>%
	group_by(fullVisitorId, date) %>%
	as_tibble()

# Classification Feature Engineering -----------------------------------------------------

tri <- 1:nrow(train_class)
y <- train_class %>%
	group_by(fullVisitorId) %>%
	summarise(target_class = sum(ifelse(transactionRevenue == 0, 0, 1))) %>%
	mutate(target_class = ifelse(target_class > 1, 1, target_class)) %>%
	.$target_class



train_ids <- train_class$fullVisitorId %>% unique()
test_ids <-  test_class$fullVisitorId %>% unique()

tmp <- outersect(colnames(train_class),colnames(test_class))

train_class <- train_class %>% ungroup() %>% as_tibble()

tr_te <- train_class[ , -which(names(train_class) %in% tmp)] %>%
	rbind(test_class %>% ungroup()) %>%
	data.table()

rm(train_class, test_class, tmp)
gc()

tr_te[, weekend := ifelse((date %>% wday(label = TRUE, abbr = FALSE)) %in% c("Saturday", "Sunday"),1,0)]
tr_te[, date := NULL]
tr_te[, browser := ifelse(browser %in% c("Safari", "Firefox"), "mainstream", ifelse(browser == "Chrome", 'Chrome', "Other"))]
tr_te[, is_chrome_the_browser := ifelse(browser == "Chrome", 1, 0) %>% as.numeric()]
tr_te[, browser := NULL]
tr_te[, source_from_googleplex := ifelse(source == 'mail.googleplex.com', 1, 0) %>% as.numeric()]
tr_te[, source_from_youtube := ifelse(source == 'youtube.com', 1, 0) %>% as.numeric()]
tr_te[, source := NULL]
tr_te[, is_medium_referral := ifelse(medium == 'referral', 1, 0) %>% as.numeric()]
tr_te[, medium := NULL]
tr_te[, is_device_desktop := ifelse(deviceCategory == 'desktop', 1, 0) %>% as.numeric()]
tr_te[, is_device_macbook := is_device_desktop*ifelse(operatingSystem == "Macintosh", 1, 0)]
tr_te[, windows_desktop := is_device_desktop*ifelse(operatingSystem == 'Windows', 1, 0)]
tr_te[, is_device_chromebook := ifelse(operatingSystem == "Chrome OS", 1, 0)]
tr_te[, is_device_linux := ifelse(operatingSystem == "Linux", 1, 0)]
tr_te[, is_phone_ios := ifelse(operatingSystem == "iOS", 1, 0)]
tr_te[, is_phone_android := ifelse(operatingSystem == "Android", 1, 0)]
tr_te[, operatingSystem := NULL]
tr_te[, deviceCategory := NULL]
tr_te[, single_visit := ifelse(visitNumber == 1,1,0) ]
tr_te[, hits_ratio := as.numeric(hits)/as.numeric(pageviews)]

tr_te[,which(unlist(lapply(tr_te, function(x)!all(is.na(x))))),with=F]

all_ids <- tr_te$fullVisitorId
tr_te <- tr_te %>%
	select(-visitId,-visitStartTime, -sessionId, -fullVisitorId) %>%
	numerise_data() %>%
	data.table()
tr_te$fullVisitorId <- all_ids %>% as.character()

tr_te <- tr_te %>%
	group_by(fullVisitorId) %>%
	summarise_all(fn)

tr_te[is.na(tr_te)] <- 0

# Classification Prepare Data -----------------------------------------------------------


validation_ids <- sample(train_ids, floor(length(train_ids)*0.1))

test_rf <- tr_te %>%
	ungroup() %>%
	filter(fullVisitorId %in% test_ids) %>%
	dplyr::select(-fullVisitorId) %>%
	as_tibble()

train_rf <- tr_te %>%
	ungroup() %>%
	filter(fullVisitorId %in% train_ids) %>%
	as_tibble()

y[is.na(y)] <- 0
train_rf$target <- y

validation_set <- train_rf %>%
	filter(fullVisitorId %in% validation_ids) %>%
	dplyr::select(-fullVisitorId) %>%
	as_tibble()

rm(tr_te);gc()

train_rf <- train_rf %>%
	filter(!(fullVisitorId %in% validation_ids)) %>%
	dplyr::select(-fullVisitorId)

train_rf_bal <- train_rf %>%
	group_by(target) %>%
	sample_n(floor(length(y)/2), replace = TRUE) %>%
	ungroup()


# Classification XGB ------------------------------------------------------

y <- train_rf$target
train_xgb <- xgb.DMatrix(data = train_rf %>% select(-target) %>% as.matrix(),
												 label = y)

train_xgb_bal <- xgb.DMatrix(data = train_rf_bal %>% select(-target) %>% as.matrix(),
														 label = train_rf_bal$target)
val_xgb <- xgb.DMatrix(data = validation_set %>% select(-target) %>% as.matrix(),
											 label = validation_set$target)
test_xgb  <- xgb.DMatrix(data = test_rf %>% as.matrix())

p <- list(objective = "binary:logistic",
					booster = "gbtree",
					eval_metric = "auc",
					nthread = 4,
					eta = 0.002,
					max_depth = 6,
					min_child_weight = 30,
					gamma = 0,
					subsample = 0.85,
					colsample_bytree = 0.7,
					colsample_bylevel = 0.632,
					alpha = 0,
					lambda = 0,
					nrounds = 1e5)

m_xgb <- xgb.train(p, train_xgb, p$nrounds, list(val = val_xgb), print_every_n = 50, early_stopping_rounds = 300)
m_xgb_bal <- xgb.train(p, train_xgb_bal, p$nrounds, list(val = val_xgb), print_every_n = 50, early_stopping_rounds = 300)


# Classification Cross Validation --------------------------------------------------------

# prediction_xgb <- predict(m_xgb,val_xgb, type = "prob")
# prediction_xgb_bal <- predict(m_xgb_bal,val_xgb, type = "prob")
#
# confusionMatrix(data = as.factor(as.numeric(ifelse(prediction_xgb > 0.5, 1, 0))), as.factor(validation_set$target))
# confusionMatrix(data = as.factor(as.numeric(ifelse(prediction_xgb_bal > 0.5, 1, 0))), as.factor(validation_set$target))
#
# prediction_xgb <- (prediction_xgb + prediction_xgb_bal)/2
# confusionMatrix(data = as.factor(as.numeric(ifelse(prediction_xgb > 0.5, 1, 0))), as.factor(validation_set$target))


# Classification Prediction --------------------------------------------------------------
prediction_xgb <- predict(m_xgb,test_xgb, type = "prob")
prediction_xgb_bal <- predict(m_xgb_bal,test_xgb, type = "prob")

prediction_xgb <- (prediction_xgb + prediction_xgb_bal)/2

prediction_final <- prediction_xgb

prediction_class <- ifelse(prediction_final >= 0.5, 1, 0)


# Regression Data Sets ----------------------------------------------------

rm(m_xgb,m_xgb_bal,sub,test_rf,train_rf,train_rf_bal,validation_set);gc()

train_reg <- train %>%
	mutate(date = ymd(date)) %>%
	group_by(fullVisitorId,date) %>%
	mutate(target_reg = as.numeric(transactionRevenue)) %>%
	as_tibble()

test_reg <- test %>%
	mutate(date = ymd(date)) %>%
	group_by(fullVisitorId, date) %>%
	as_tibble()



# Regression Feature Engineering ------------------------------------------

# Set target
tri <- 1:nrow(train_reg)
y <- train_reg %>%
	group_by(fullVisitorId) %>%
	summarise(target_reg = log(sum(as.numeric(transactionRevenue)))) %>%
	filter(target_reg != 0) %>%
	.$target_reg

# Separate Testand Train Ids
train_ids <- train_reg %>%
	group_by(fullVisitorId) %>%
	summarise(target_reg = log(sum(as.numeric(transactionRevenue)))) %>%
	filter(target_reg != 0) %>%
	.$fullVisitorId %>% unique()

test_ids <-  test_reg$fullVisitorId %>% unique()

# Build Features Data Set
tmp <- outersect(colnames(train_reg),colnames(test_reg))
train_reg <- train_reg %>% ungroup() %>% as_tibble()
tr_te <- train_reg[ , -which(names(train_reg) %in% tmp)] %>%
	rbind(test_reg %>% ungroup()) %>%
	data.table()

rm(train_reg, test_reg, tmp); gc()


# Building Features
tr_te[, weekend := ifelse((date %>% wday(label = TRUE, abbr = FALSE)) %in% c("Saturday", "Sunday"),1,0)]
tr_te[, date := NULL]
tr_te[, browser := ifelse(browser %in% c("Safari", "Firefox"), "mainstream", ifelse(browser == "Chrome", 'Chrome', "Other"))]
tr_te[, is_chrome_the_browser := ifelse(browser == "Chrome", 1, 0) %>% as.numeric()]
tr_te[, browser := NULL]
tr_te[, source_from_googleplex := ifelse(source == 'mail.googleplex.com', 1, 0) %>% as.numeric()]
tr_te[, source_from_youtube := ifelse(source == 'youtube.com', 1, 0) %>% as.numeric()]
tr_te[, source := NULL]
tr_te[, is_medium_referral := ifelse(medium == 'referral', 1, 0) %>% as.numeric()]
tr_te[, medium := NULL]
tr_te[, is_device_desktop := ifelse(deviceCategory == 'desktop', 1, 0) %>% as.numeric()]
tr_te[, is_device_macbook := is_device_desktop*ifelse(operatingSystem == "Macintosh", 1, 0)]
tr_te[, windows_desktop := is_device_desktop*ifelse(operatingSystem == 'Windows', 1, 0)]
tr_te[, is_device_chromebook := ifelse(operatingSystem == "Chrome OS", 1, 0)]
tr_te[, is_device_linux := ifelse(operatingSystem == "Linux", 1, 0)]
tr_te[, is_phone_ios := ifelse(operatingSystem == "iOS", 1, 0)]
tr_te[, is_phone_android := ifelse(operatingSystem == "Android", 1, 0)]
tr_te[, operatingSystem := NULL]
tr_te[, deviceCategory := NULL]
tr_te[, single_visit := ifelse(visitNumber == 1,1,0) ]
tr_te[, hits_ratio := as.numeric(hits)/as.numeric(pageviews)]
tr_te[,which(unlist(lapply(tr_te, function(x)!all(is.na(x))))),with=F]


# Cleaning up
tr_te[,which(unlist(lapply(tr_te, function(x)!all(is.na(x))))),with=F]

all_ids <- tr_te$fullVisitorId

tr_te <- tr_te %>%
	select(-visitId,-visitStartTime, -sessionId, -fullVisitorId) %>%
	numerise_data() %>%
	data.table()
tr_te$fullVisitorId <- all_ids %>% as.character()

tr_te <- tr_te %>%
	group_by(fullVisitorId) %>%
	summarise_all(fn)

tr_te[is.na(tr_te)] <- 0



# Regression Preparing Data ---------------------------------------

#Preparing Data
validation_ids <- sample(train_ids, floor(length(train_ids)*0.1))

test_rf <- tr_te %>%
	ungroup() %>%
	filter(fullVisitorId %in% test_ids) %>%
	dplyr::select(-fullVisitorId) %>%
	as_tibble()

train_rf <- tr_te %>%
	ungroup() %>%
	filter(fullVisitorId %in% train_ids) %>%
	as_tibble()

y[is.na(y)] <- 0
train_rf$target <- y

train_rf <- train_rf %>% filter(!is.infinite(target))

validation_set <- train_rf %>%
	filter(fullVisitorId %in% validation_ids) %>%
	dplyr::select(-fullVisitorId) %>%
	as_tibble()

rm(tr_te);gc()

train_rf <- train_rf %>%
	filter(!(fullVisitorId %in% validation_ids)) %>%
	dplyr::select(-fullVisitorId)

# Regression XGB ----------------------------------------------------------

y <- train_rf$target
train_xgb <- xgb.DMatrix(data = train_rf %>% select(-target) %>% as.matrix(), label = y)
val_xgb <- xgb.DMatrix(data = validation_set %>% select(-target) %>% as.matrix(), label = validation_set$target)
test_xgb  <- xgb.DMatrix(data = test_rf %>% as.matrix())

p <- list(objective = "reg:linear",
					booster = "gbtree",
					eval_metric = "rmse",
					nthread = 4,
					eta = 0.0025,
					max_depth = 6,
					min_child_weight = 30,
					gamma = 0,
					subsample = 0.85,
					colsample_bytree = 0.7,
					colsample_bylevel = 0.632,
					alpha = 0,
					lambda = 0,
					nrounds = 5000)

m_xgb <- xgb.train(p, train_xgb, p$nrounds, list(val = val_xgb), print_every_n = 50, early_stopping_rounds = 300)

# Regression Cross Validation ---------------------------------------------
prediction_xgb <- predict(m_xgb, val_xgb) %>% c()
plot(prediction_xgb,validation_set$target)
sum((prediction_xgb-validation_set$target)^2)

# Submission --------------------------------------------------------------


prediction_reg <- predict(m_xgb, test_xgb)   %>% c()


sub <- read_csv("data/sample_submission.csv")
sub$PredictedLogRevenue <- (prediction_class)*prediction_reg %>% c()

sub %>%
	fwrite("data/submission.csv")

toc()
