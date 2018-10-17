# Libraries & Functions ---------------------------------------------------



library(tidyverse)
library(data.table)
library(jsonlite)
library(lubridate)
library(moments)

library(tictoc)

library(caret)
library(xgboost)
library(randomForest)

library(viridis)
library(harrypotter)

library(lime)



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

# Resampling --------------------------------------------------------------

y <- train_class[["target_class"]]
y[is.na(y)] <- 0
train_class[["target_class"]] <- y
n <- nrow(train_class)

n_pos <- sum(train_class$target_class==1)
n_tot <- nrow(train_class)
train_positive <- train_class %>%
	filter(target_class == 1) %>%
	sample_n(size = floor((n_tot - n_pos)/(2*n_pos)), replace = TRUE)

train_class <- rbind(train_positive, train_class) %>%
	data.table()

train_class <- train_class[, .SD[sample(.N, n)]] %>% as_tibble()

rm(y,n, n_pos,n_tot,train_positive); gc()
# Feature Engineering -----------------------------------------------------

tri <- 1:nrow(train_class)
y <- train_class %>%
	group_by(fullVisitorId) %>%
	summarise(target_class = sum(ifelse(transactionRevenue == 0, 0, 1))) %>%
	mutate(target_class = ifelse(target_class > 1, 1, target_class)) %>%
	.$target_class



train_ids <- train_class$fullVisitorId %>% unique()
test_ids <-  test_class$fullVisitorId %>% unique()

outersect <- function(x, y) {
	sort(c(setdiff(x, y),
				 setdiff(y, x)))
}

tmp <- outersect(colnames(train_class),colnames(test_class))


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

tr_te[,which(unlist(lapply(tr_te, function(x)!all(is.na(x))))),with=F]

fn <- funs(mean,
					 sd,
					 # min,
					 # max,
					 # sum,
					 n_distinct,
					 # kurtosis,
					 # skewness,
					 .args = list(na.rm = TRUE))

all_ids <- tr_te$fullVisitorId
tr_te <- tr_te %>%
	select(-visitId,-visitStartTime, -sessionId, -fullVisitorId) %>%
	numerise_data() %>%
	data.table()
tr_te$fullVisitorId <- all_ids %>% as.character()

# tr_te[, lapply(.SD, mean, na.rm=TRUE), by=fullVisitorId ]

tr_te <- tr_te %>%
	group_by(fullVisitorId) %>%
	summarise_all(fn)

tr_te[is.na(tr_te)] <- 0


# Random Forest -----------------------------------------------------------

#Preparing Data
trte_rf <- tr_te %>%
	ungroup() %>%
	dplyr::select(-fullVisitorId) %>%
	as_tibble()

train_rf <- tr_te %>%
	ungroup() %>%
	filter(fullVisitorId %in% train_ids) %>%
	dplyr::select(-fullVisitorId) %>%
	as_tibble()

test_rf <- tr_te %>%
	ungroup() %>%
	filter(fullVisitorId %in% test_ids) %>%
	dplyr::select(-fullVisitorId) %>%
	as_tibble()

# train_rf <- trte_rf[tri,]
# test_rf <-  trte_rf[-tri,]

y[is.na(y)] <- 0
train_rf$target <- y

rf_model <- randomForest(factor(target) ~.,
						 data = train_rf)

# Show model error
plot(rf_model, ylim=c(0,0.36))
legend('topright', colnames(rf_model$err.rate), col=1:3, fill=1:3)

# Get importance
importance    <- importance(rf_model)
varImportance <- data.frame(Variables = row.names(importance),
														Importance = round(importance[ ,'MeanDecreaseGini'],2))

# Create a rank variable based on importance
rankImportance <- varImportance %>%
	mutate(Rank = paste0('#',dense_rank(desc(Importance))))

# Use ggplot2 to visualize the relative importance of variables
ggplot(rankImportance, aes(x = reorder(Variables, Importance),
													 y = Importance, fill = Importance)) +
	geom_bar(stat='identity') +
	labs(x = 'Variables') +
	coord_flip() +
	scale_fill_viridis() +
	theme_minimal()

prediction <- predict(rf_model, test_rf) %>% c()
sub <- read_csv("data/sample_submission.csv")

sub$PredictedLogRevenue <- (prediction-1)*median_revenue %>% c()

rf_model
