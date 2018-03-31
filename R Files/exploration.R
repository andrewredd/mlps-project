library(readr)
library(dplyr)
library(stringr)
library(mice)

setwd("C:/Users/andre/GoogleDrive/MLPSProject")

# load data
data = read_tsv("Data/train.tsv")

# replace no item descriptions with nas
data$item_description[data$item_description == "No description yet"] = NA

# which columns have nas
apply(data, 2, function(x) any(is.na(x)))

# total number of nas
data %>% is.na() %>% colSums()

# View item descriptions by category
data %>% filter(is.na(item_description)) %>%  group_by(category_name) %>% count() 

# View category structure
data %>% 
  select(category_name) %>% 
  unique() %>% 
  mutate(num = str_count(category_name, pattern = "/")) %>% 
  filter(num != 2) %>% View()
