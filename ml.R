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
library(randomForest)
library(h2o)

library(viridis)
library(harrypotter)




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
					 # sd,
					 # # min,
					 # # max,
					 # # sum,
					 n_distinct,
					 # kurtosis,
					 # skewness,
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

# Classification Random Forest -----------------------------------------------------------

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
	sample_n(floor(length(y)/2), replace = TRUE)


# Actual Training
rf_model <- randomForest(factor(target) ~.,
						 data = train_rf)

rf_model_bal <- randomForest(factor(target) ~.,
												 data = train_rf_bal)



# Classification H2O ------------------------------------------------------
h2o.init()
train_h2o <- as.h2o(train_rf %>% mutate(target = as.factor(target)))
train_h2o_bal <- as.h2o(train_rf_bal %>% ungroup() %>%  mutate(target = as.factor(target)))
test_h2o <- as.h2o(test_rf )
validation_set_h2o <- as.h2o(validation_set %>% mutate(target = as.factor(target)))

model_h2o <- h2o.deeplearning(x = setdiff(names(train_rf), c("target")),
															y = "target",
															training_frame = train_h2o,
															standardize = TRUE,         # standardize data
															hidden = c(50, 25, 10),       # 2 layers of 00 nodes each
															rate = 0.005,                # learning rate
															epochs = 100,                # iterations/runs over data
															seed = 666                 # reproducability seed
)

model_h2o_bal <- h2o.deeplearning(x = setdiff(names(train_rf_bal), c("target")),
															y = "target",
															training_frame = train_h2o,
															standardize = TRUE,         # standardize data
															hidden = c(50, 25, 10),       # 2 layers of 00 nodes each
															rate = 0.005,                # learning rate
															epochs = 100,                # iterations/runs over data
															seed = 666                 # reproducability seed
)

# Classification Cross Validation --------------------------------------------------------
prediction <- 1 - predict(rf_model, validation_set, type="prob")[,1] %>% c()
prediction_bal <- 1 - predict(rf_model_bal, validation_set, type="prob")[,1] %>% c()

h2o.predictions <- as.data.frame(h2o.predict(model_h2o, validation_set_h2o))[[3]]
h2o.predictions_bal <- as.data.frame(h2o.predict(model_h2o_bal, validation_set_h2o))[[3]]

h2o.prediction <- (h2o.predictions + h2o.predictions_bal)/2
h2o.prediction <- ifelse(h2o.prediction >= 0.5, 1, 0) %>% as.factor()

confusionMatrix(data = as.factor(as.numeric(ifelse(h2o.predictions > 0.5, 1, 0))), as.factor(validation_set$target))
confusionMatrix(data = as.factor(as.numeric(ifelse(h2o.predictions_bal > 0.5, 1, 0))), as.factor(validation_set$target))
confusionMatrix(data = as.factor(h2o.prediction), as.factor(validation_set$target))

prediction_rf_prob <- (prediction + prediction_bal)/2
prediction_rf <- ifelse(prediction_rf_prob >= 0.5, 1, 0) %>% as.factor()


confusionMatrix(data = as.factor(as.numeric(ifelse(prediction > 0.5, 1, 0))), as.factor(validation_set$target))
confusionMatrix(data = as.factor(as.numeric(ifelse(prediction_bal > 0.5, 1, 0))), as.factor(validation_set$target))
confusionMatrix(data = prediction_rf, as.factor(validation_set$target))

prediction_final <- ifelse((h2o.predictions + prediction_rf_prob)/2 >= 0.5, 1, 0) %>% as.factor()
confusionMatrix(data = prediction_final, as.factor(validation_set$target))


# Classification Prediction --------------------------------------------------------------
prediction <- 1 - predict(rf_model, test_rf, type="prob")[,1] %>% c()
prediction_bal <- 1 - predict(rf_model_bal, test_rf, type="prob")[,1] %>% c()

h2o.predictions <- as.data.frame(h2o.predict(model_h2o, test_h2o))[[3]]
h2o.predictions_bal <- as.data.frame(h2o.predict(model_h2o_bal, test_h2o))[[3]]

prediction_h20 <- (h2o.predictions + h2o.predictions_bal)/2
prediction_rf <- (prediction + prediction_bal)/2

prediction_final <- (prediction_h20 + prediction_rf)/2

prediction_class <- ifelse(prediction_final >= 0.5, 1, 0)


# Regression Data Sets ----------------------------------------------------

rm(rf_model,rf_model_bal,sub,test_rf,train_rf,train_rf_bal,validation_set);gc()

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



# Regression Training Random Forest ---------------------------------------

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


# Actual Training
rf_model <- randomForest((target) ~.,
												 data = train_rf)

varImportance <- rf_model$importance

varImportance <- data.frame(Variables = rownames(varImportance),
														Importance = as_tibble(varImportance)$IncNodePurity)

variables <- varImportance %>%
	arrange(-Importance) %>%
	top_n(20, Importance) %>%
	.$Variables %>%
	paste(collapse="+")

formula <- paste("target ~ ", variables, sep = "") %>% as.formula()

lm_model <- lm(formula, data = train_rf)



# Regression H2O ---------------------------------------------------------

train_h2o <- as.h2o(train_rf)
test_h2o <- as.h2o(test_rf)
validation_set_h2o <- as.h2o(validation_set)

model_h2o <- h2o.deeplearning(x = setdiff(names(train_rf), c("target")),
															y = "target",
															training_frame = train_h2o,
															standardize = TRUE,         # standardize data
															hidden = c(50, 25, 10),       # 2 layers of 00 nodes each
															rate = 0.005,                # learning rate
															epochs = 100,                # iterations/runs over data
															seed = 666                 # reproducability seed
)


# Regression Cross Validation ---------------------------------------------

prediction_rf <- predict(rf_model, validation_set) %>% c()
plot(prediction_rf,validation_set$target)
sum((prediction_rf-validation_set$target)^2)


prediction_lm <- predict(lm_model, validation_set) %>% c()
plot(prediction_lm,validation_set$target)
sum((prediction_lm-validation_set$target)^2)


h2o.predictions <- as.data.frame(h2o.predict(model_h2o, validation_set_h2o))[[1]]
plot(h2o.predictions,validation_set$target)
sum((h2o.predictions-validation_set$target)^2)

plot(prediction_rf,prediction_lm)

prediction <- (prediction_rf + prediction_lm + h2o.predictions)/3
plot(prediction,validation_set$target)
sum((prediction-validation_set$target)^2)
# Submission --------------------------------------------------------------


prediction_reg_rf  <- predict(rf_model, test_rf) %>% c()
prediction_reg_lm  <- predict(lm_model, test_rf) %>% c()
prediction_reg_h2o <- as.data.frame(h2o.predict(model_h2o, test_h2o))[[1]]

prediction_reg <- prediction_reg_h2o
prediction_reg <- (prediction_reg_rf + prediction_reg_lm)/2



sub <- read_csv("data/sample_submission.csv")
sub$PredictedLogRevenue <- (prediction_class)*prediction_reg %>% c()

sub %>%
	fwrite("data/submission.csv")


toc()
