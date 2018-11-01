
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

get_folds <- function(data, group, v = 5) {
	group_folds <- group_vfold_cv(data[group], group, v = v)
	list.zip(tr = tr_ind <- map(group_folds$splits, ~ .x$in_id),
					 val = val_ind <- map(group_folds$splits, ~ setdiff(1:nrow(data), .x$in_id)))
}

create_time_fea <- function(fun = lag, n = 1){
	tr_tmp <- tr_te %>% dplyr::slice(tri) %>% data.table()
	te_tmp <- tr_te %>% dplyr::slice(-tri) %>% data.table()

	tr_tmp[, time_var := (date - fun(date, n)) %>% as.integer()/3600]
	te_tmp[, time_var := (date - fun(date, n)) %>% as.integer()/3600]

	output <- c(tr_tmp$time_var, te_tmp$time_var)

	rm(tr_tmp, te_tmp)
#
#
# 	c((tr_te$date[tri] - (select(tr_te, fullVisitorId, date) %>%
# 													dplyr::slice(tri) %>%
# 													group_by(fullVisitorId) %>%
# 													mutate(time_var = fun(date, n)) %$%
# 													time_var)) %>% as.integer / 3600,
# 		(tr_te$date[-tri] - (select(tr_te, fullVisitorId, date) %>%
# 												 	dplyr::slice(-tri) %>%
# 												 	group_by(fullVisitorId) %>%
# 												 	mutate(time_var = fun(date, n)) %$%
# 												 	time_var)) %>% as.integer / 3600
# 	)
	return(output)
}
has_many_values <- function(x) n_distinct(x) > 1

