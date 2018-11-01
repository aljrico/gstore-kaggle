# Retrieve Data -----------------------------------------------------------

train <- read_csv("data/train.csv") %>%
	# sample_n(5e4)  %>%
	flatten() %>%
	data.table()

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
train <- train[,which(unlist(lapply(train, function(x)!all(is.na(x))))),with=F]
test <- test[,which(unlist(lapply(test, function(x)!all(is.na(x))))),with=F]


# Formatting --------------------------------------------------------------

train[, date := ymd(date)]
test[,  date := ymd(date)]


# Remove Outliers ---------------------------------------------------------

train %>% filter(country != 'Anguilla') %>% data.table()


