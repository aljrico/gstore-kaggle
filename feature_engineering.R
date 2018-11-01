feature_engineering <- function(tr_te, models = NULL, tri = 0){
	tr_te <- data.table(tr_te)
		cat("Feature Engineering: \n")
#
# 	tr_te <- tr_te %>%
# 		arrange(date) %>%
# 		group_by(date,fullVisitorId) %>%
# 		mutate(number_previous_visits = cumsum(n())) %>%
# 		data.table()


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
	tr_te[, path_depth := str_count(referralPath, '/')]
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

	tr_te %<>% data.table()

	# Group Features
	# 	cat("... Group Features ... \n")
	# for (grp in c("year", "wday", "hour")) {
	# 	col <- paste0(grp, "_user_cnt")
	# 	# tr_te %<>%
	# 	# 	group_by_(grp) %>%
	# 	# 	mutate(!!col := n_distinct(fullVisitorId)) %>%
	# 	# 	ungroup()
	#
	# 	tr_te[, (deparse(col)) := n_distinct(fullVisitorId), by = get(grp)]
	# }


		all_ids <- tr_te$fullVisitorId

		tr_te <- tr_te %>%
			select(-visitId, -sessionId, -fullVisitorId) %>%
			numerise_data() %>%
			as_tibble()


	# 	cat("... ... Summarising Group Features ... \n ")
	# fn <- funs(mean, median, var, min, max, sum, n_distinct, .args = list(na.rm = TRUE))
	# for (grp in c("browser", "city", "country", "domain_site",
	# 							"path_depth", "source_depth", "visitNumber")) {
	# 	df <- paste0("sum_by_", grp)
	# 	s <- paste0("_", grp)
	# 	tr_te %<>%
	# 		left_join(assign(df, tr_te %>%
	# 										 	select_(grp, "pageviews") %>%
	# 										 	group_by_(grp) %>%
	# 										 	summarise_all(fn)),  by = grp, suffix = c("", s))
	# }

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

	# redundant_columns <- whichAreBijection(tr_te, verbose=TRUE)
	# if(length(redundant_columns > 0)) tr_te <- subset(tr_te,select = -c(redundant_columns)) %>% as_tibble()


	# tr_te$fullVisitorId <- all_ids %>% as.character()
	tr_te[is.na(tr_te)] <- 0
	return(tr_te)
}

# set.seed(0)
# df <- data.frame(V1 = sample(c("a", "b", "c"), 11, TRUE),
# 								 V2 = sample(c("2016", "2017", "2018"), 11, TRUE),
# 								 V3 = sample(seq(1:3), 11, TRUE),
# 								 V4 = sample(seq(1:3), 11, TRUE),
# 								 Id = sample(seq(1:5), 11, TRUE))
#
# for (grp in c("V1", "V2", "V3", "V4")) {
# 	col <- paste0(grp, "_user_cnt")
# 	df %<>%
# 		group_by_(grp) %>%
# 		mutate(!!col := n_distinct(Id)) %>%
# 		ungroup()
# }
#
# DT <- data.table(df)
# for (grp in c("V1", "V2", "V3", "V4")) {
# 	col <- paste0(grp, "_user_cnt")
# 	DT[, (deparse(col)) := n_distinct(Id), by = get(grp)]
# }
