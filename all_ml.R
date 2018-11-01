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


# Workflow ----------------------------------------------------------------
tuning_fast <- TRUE
source("preprocessing.R");gc()
source("classification_session.R")
source("classification_user.R")
source("regression_session.R")
class_session <- classification_session(train,test, fast = tuning_fast);gc()
xgb.importance(model = model_class_sess_xgb) %>% as_tibble() %>% top_n(25, Gain) %>%
	ggplot(aes(x = reorder(Feature, Gain), y = Gain)) +
	geom_col() +
	coord_flip()
# class_user <- classification_user(train,test, fast = tuning_fast, models_to_feature = NULL);gc()
reg_session <- regression_session(train,test, fast = tuning_fast, models = model_class_sess_xgb);gc()
xgb.importance(model = model_reg_sess_xgb) %>% as_tibble() %>% top_n(25, Gain) %>%
	ggplot(aes(x = reorder(Feature, Gain), y = Gain)) +
	geom_col() +
	coord_flip()

submit <- . %>%
	as_tibble() %>%
	set_names("y") %>%
	mutate(y = ifelse(y < 0, 0, exp(y))) %>%
	bind_cols(id) %>%
	group_by(fullVisitorId) %>%
	summarise(y = log1p(1 + sum(y))) %>%
	right_join(
		read_csv("data/sample_submission.csv"),
		by = "fullVisitorId") %>%
	mutate(PredictedLogRevenue = round(y, 5)) %>%
	select(-y) %>%
	write_csv("data/submission_new.csv")


id <- test$fullVisitorId %>% as_tibble() %>% set_names("fullVisitorId")
submit(class_session*reg_session)
toc()
