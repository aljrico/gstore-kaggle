library(jsonlite)
library(lubridate)
library(xgboost)
library(rsample)
library(rlist)
library(tidyverse)
library(magrittr)
library(tictoc)
tic()

set.seed(0)

#---------------------------
cat("Loading data...\n")

tr <- read_csv("data/train.csv")
te <- read_csv("data/test.csv")

#---------------------------
cat("Defining auxiliary functions...\n")

flatten_json <- . %>%
	str_c(., collapse = ",") %>%
	str_c("[", ., "]") %>%
	fromJSON(flatten = T)

parse <- . %>%
	bind_cols(flatten_json(.$device)) %>%
	bind_cols(flatten_json(.$geoNetwork)) %>%
	bind_cols(flatten_json(.$trafficSource)) %>%
	bind_cols(flatten_json(.$totals)) %>%
	select(-device, -geoNetwork, -trafficSource, -totals)

is_na_val <- function(x) x %in% c("not available in demo dataset", "(not provided)",
																	"(not set)", "<NA>", "unknown.unknown",  "(none)")

has_many_values <- function(x) n_distinct(x) > 1

get_folds <- function(data, group, v = 5) {
	group_folds <- group_vfold_cv(data[group], group, v = v)
	list.zip(tr = tr_ind <- map(group_folds$splits, ~ .x$in_id),
					 val = val_ind <- map(group_folds$splits, ~ setdiff(1:nrow(data), .x$in_id)))
}

create_time_fea <- function(fun = lag, n = 1)
	c((tr_te$date[tri] - (select(tr_te, fullVisitorId, date) %>%
													slice(tri) %>%
													group_by(fullVisitorId) %>%
													mutate(time_var = fun(date, n)) %$%
													time_var)) %>% as.integer / 3600,
		(tr_te$date[-tri] - (select(tr_te, fullVisitorId, date) %>%
												 	slice(-tri) %>%
												 	group_by(fullVisitorId) %>%
												 	mutate(time_var = fun(date, n)) %$%
												 	time_var)) %>% as.integer / 3600
	)

#---------------------------
cat("Basic preprocessing...\n")

tr <- parse(tr)
te <- parse(te)

y <- log1p(as.numeric(tr$transactionRevenue))
y[is.na(y)] <- 0

tr$transactionRevenue <- NULL
tr$campaignCode <- NULL
tri <- 1:nrow(tr)

tr_id <- tr$fullVisitorId
te_id <- te$fullVisitorId
#idx <- ymd(tr$date) < ymd("20170601")

tr_val_ind <- get_folds(tr, "fullVisitorId", 5)

tr_te <- tr %>%
	bind_rows(te) %>%
	select_if(has_many_values) %>%
	mutate_all(funs(ifelse(is_na_val(.), NA, .))) %>%
	mutate(pageviews = ifelse(is.na(pageviews), 0L, as.integer(pageviews)),
				 visitNumber =  visitNumber,
				 newVisits = ifelse(newVisits == "1", 1L, 0L),
				 bounces = ifelse(is.na(bounces), 0L, 1L),
				 isMobile = ifelse(isMobile, 1L, 0L),
				 adwordsClickInfo.isVideoAd = ifelse(is.na(adwordsClickInfo.isVideoAd), 0L, 1L),
				 isTrueDirect = ifelse(is.na(isTrueDirect), 0L, 1L),
				 browser_dev = str_c(browser, "_", deviceCategory),
				 browser_os = str_c(browser, "_", operatingSystem),
				 browser_chan = str_c(browser,  "_", channelGrouping),
				 campaign_medium = str_c(campaign, "_", medium),
				 chan_os = str_c(operatingSystem, "_", channelGrouping),
				 country_adcontent = str_c(country, "_", adContent),
				 country_medium = str_c(country, "_", medium),
				 country_source = str_c(country, "_", source),
				 dev_chan = str_c(deviceCategory, "_", channelGrouping),
				 date = as_datetime(visitStartTime),
				 year = year(date),
				 wday = wday(date),
				 hour = hour(date))

for (i in c("city", "continent", "country", "metro", "networkDomain", "region", "subContinent"))
	for (j in c("browser", "deviceCategory", "operatingSystem", "source"))
		tr_te[str_c(i, "_", j)] <- str_c(tr_te[[i]], tr_te[[j]], sep = "_")

#---------------------------
cat("Adding time features...\n")

for (i in 1:5) {
	tr_te[str_c("next_sess", i)] <- create_time_fea(lag, i)
	tr_te[str_c("prev_sess", i)] <- create_time_fea(lead, i)
}

#---------------------------
cat("Creating group features...\n")

for (grp in c("year", "wday", "hour")) {
	col <- paste0(grp, "_user_cnt")
	tr_te %<>%
		group_by_(grp) %>%
		mutate(!!col := n_distinct(fullVisitorId)) %>%
		ungroup()
}

fn <- funs(mean, median, var, min, max, sum, n_distinct, .args = list(na.rm = TRUE))
for (grp in c("browser", "city", "country", "networkDomain",
							"referralPath", "source", "visitNumber")) {
	df <- paste0("sum_by_", grp)
	s <- paste0("_", grp)
	tr_te %<>%
		left_join(assign(df, tr_te %>%
										 	select_(grp, "pageviews") %>%
										 	group_by_(grp) %>%
										 	summarise_all(fn)),  by = grp, suffix = c("", s))
}

tr_te %<>%
	select(-date, -fullVisitorId, -visitId, -sessionId, -visitStartTime, -hits) %>%
	mutate_if(is.character, funs(factor(.) %>% as.integer)) %>%
	select_if(has_many_values)

rm(tr, te, grp, col, flatten_json, parse,
	 has_many_values, is_na_val, fn, df, s)
gc()

#---------------------------
cat("Training model: session level...\n")

dtest <- xgb.DMatrix(data = data.matrix(tr_te[-tri, ]))
pred_tr <- rep(0, length(tri))
pred_te <- 0

for (i in seq_along(tr_val_ind)) {
	cat("Group fold:", i, "\n")

	tr_ind <- tr_val_ind[[i]]$tr
	val_ind <- tr_val_ind[[i]]$val

	dtrain <- xgb.DMatrix(data = data.matrix(tr_te[tr_ind, ]), label = y[tr_ind])
	dval <- xgb.DMatrix(data = data.matrix(tr_te[val_ind, ]), label = y[val_ind])

	p <- list(objective = "reg:linear",
						booster = "gbtree",
						eval_metric = "rmse",
						nthread = 4,
						eta = 0.075,
						max_depth = 8,
						min_child_weight = 100,
						gamma = 5,
						subsample = 1,
						colsample_bytree = 0.95,
						colsample_bylevel = 0.35,
						alpha = 25,
						lambda = 25)

	set.seed(0)
	cv <- xgb.train(p, dtrain, 1000, list(val = dval),
									print_every_n = 100, early_stopping_rounds = 100)

	pred_tr[val_ind] <- expm1(predict(cv, dval))
	pred_te <- pred_te + expm1(predict(cv, dtest))

	rm(dtrain, dval, tr_ind, val_ind, p)
	gc()
}

pred_tr <- ifelse(pred_tr < 0, 0, pred_tr)
pred_te <- ifelse(pred_te < 0, 0, pred_te / length(tr_val_ind))

cols <- colnames(tr_te)
imp <- xgb.importance(cols, model = cv) %T>%
	xgb.plot.importance(top_n = 25)

rm(dtest, cv); gc()

#---------------------------
cat("Handling data at user level...\n")

y_ul <- tibble(fullVisitorId = tr_id,
							 y = expm1(y)) %>%
	group_by(fullVisitorId) %>%
	summarise_all(funs(sum(.) %>% log1p))

tr_ul <- tr_te %>%
	slice(tri) %>%
	mutate(fullVisitorId = tr_id) %>%
	group_by(fullVisitorId) %>%
	summarise_all(funs(mean(. , na.rm = TRUE)))

tr_preds <- tr_te %>%
	slice(tri) %>%
	mutate(fullVisitorId = tr_id,
				 pred = pred_tr) %>%
	select(fullVisitorId, pred) %>%
	group_by(fullVisitorId) %>%
	mutate(pred_num = str_c("pred", 1:n())) %>%
	ungroup() %>%
	spread(pred_num, pred) %>%
	mutate(p_mean = apply(select(., starts_with("pred")), 1, mean, na.rm = TRUE),
				 p_med = apply(select(., starts_with("pred")), 1, median, na.rm = TRUE),
				 p_sd = apply(select(., starts_with("pred")), 1, sd, na.rm = TRUE),
				 p_sum = apply(select(., starts_with("pred")), 1, sum, na.rm = TRUE),
				 p_min = apply(select(., starts_with("pred")), 1, min, na.rm = TRUE),
				 p_max = apply(select(., starts_with("pred")), 1, max, na.rm = TRUE))

tr_ul %<>%
	left_join(tr_preds, by = "fullVisitorId") %>%
	left_join(y_ul, by = "fullVisitorId")

y_ul <- tr_ul$y
tr_ul$y <- NULL

rm(tr_preds); gc()

te_ul <- tr_te %>%
	slice(-tri) %>%
	mutate(fullVisitorId = te_id) %>%
	group_by(fullVisitorId) %>%
	summarise_all(funs(mean(. , na.rm = TRUE)))

te_preds <- tr_te %>%
	slice(-tri) %>%
	mutate(fullVisitorId = te_id,
				 pred = pred_te) %>%
	select(fullVisitorId, pred) %>%
	group_by(fullVisitorId) %>%
	mutate(pred_num = str_c("pred", 1:n())) %>%
	ungroup() %>%
	spread(pred_num, pred) %>%
	mutate(p_mean = apply(select(., starts_with("pred")), 1, mean, na.rm = TRUE),
				 p_med = apply(select(., starts_with("pred")), 1, median, na.rm = TRUE),
				 p_sd = apply(select(., starts_with("pred")), 1, sd, na.rm = TRUE),
				 p_sum = apply(select(., starts_with("pred")), 1, sum, na.rm = TRUE),
				 p_min = apply(select(., starts_with("pred")), 1, min, na.rm = TRUE),
				 p_max = apply(select(., starts_with("pred")), 1, max, na.rm = TRUE))

te_ul %<>%
	left_join(te_preds, by = "fullVisitorId")
te_id <- te_ul$fullVisitorId
te_ul$fullVisitorId <- NULL

rm(te_preds); gc()

tr_val_ind <- get_folds(tr_ul, "fullVisitorId", 5)
tr_ul$fullVisitorId <- NULL

cols <- intersect(names(tr_ul), names(te_ul))
tr_ul %<>% select(cols)
te_ul %<>% select(cols)

rm(tr_te, cols, tr_id, tri, y); gc()

#---------------------------
cat("Training model: user level...\n")

dtest <- xgb.DMatrix(data = data.matrix(te_ul))
rm(te_ul); gc()

pred_tr <- rep(0, nrow(tr_ul))
pred_te <- 0
err <- 0

for (i in seq_along(tr_val_ind)) {
	cat("Group fold:", i, "\n")

	tr_ind <- tr_val_ind[[i]]$tr
	val_ind <- tr_val_ind[[i]]$val

	dtrain <- xgb.DMatrix(data = data.matrix(tr_ul[tr_ind, ]), label = y_ul[tr_ind])
	dval <- xgb.DMatrix(data = data.matrix(tr_ul[val_ind, ]), label = y_ul[val_ind])

	p <- list(objective = "reg:linear",
						booster = "gbtree",
						eval_metric = "rmse",
						nthread = 4,
						eta = 0.025,
						max_depth = 7,
						min_child_weight = 5,
						gamma = 0.05,
						subsample = 0.9,
						colsample_bytree = 0.8,
						colsample_bylevel = 0.632,
						alpha = 0,
						lambda = 5)

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

#---------------------------
cat("Making submission file...\n")

tibble(pred = pred_te, fullVisitorId = te_id) %>%
	right_join(read_csv("data/sample_submission.csv"),
						 by = "fullVisitorId") %>%
	mutate(PredictedLogRevenue = round(pred, 5)) %>%
	select(-pred) %>%
	write_csv(paste0("tidy_user_xGb_", round(err, 5), ".csv"))

toc()
