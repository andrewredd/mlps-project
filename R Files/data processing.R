library(readr)
library(dplyr)
library(stringr)
library(purrr)
library(text2vec)
library(Matrix)

setwd("C:/Users/andre/OneDrive/School/CMU/Spring 2018/Machine Learning for Problem Solving/Project/mlps-project")

# load data
data = read_tsv("Data/train.tsv")

data = data %>% filter(!is.na(price))

# replace no item descriptions with nas
data$item_description[data$item_description == "No description yet"] = NA
data$item_description[is.na(data$item_description)] = "NA"
data$brand_name[is.na(data$brand_name)] = "NA"
data$category_name[is.na(data$category_name)] = "NA/NA/NA"

# parse out categories
data = data %>% 
  mutate(category_list = category_name %>% strsplit(split = "/"),
         category_name_1 = map_chr(category_list, ~ .x[1]),
         category_name_2 = map_chr(category_list, ~ .x[2]),
         category_name_3 = map_chr(category_list, ~ .x[-(1:2)] %>% paste(collapse = "/")))

write_tsv(data %>% select(-category_list), "Data/train_clean.tsv")

data = read_tsv("Data/train_clean.tsv")

set.seed(95828)

it_train = itoken(data$item_description,
                  preprocessor=tolower,
                  tokenizer=word_tokenizer,
                  ids = data$train_id,
                  progressbar = FALSE)

stop_words = c("na", "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "arent", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "cant", "cannot", "could", "couldnt", "did", "didnt", "do", "does", "doesnt", "doing", "dont", "down", "during", "each", "few", "for", "from", "further", "had", "hadnt", "has", "hasnt", "have", "havent", "having", "he", "hed", "hell", "hes", "her", "here", "heres", "hers", "herself", "him", "himself", "his", "how", "hows", "i", "id", "ill", "im", "ive", "if", "in", "into", "is", "isnt", "it", "its", "its", "itself", "lets", "me", "more", "most", "mustnt", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shant", "she", "shed", "shell", "shes", "should", "shouldnt", "so", "some", "such", "than", "that", "thats", "the", "their", "theirs", "them", "themselves", "then", "there", "theres", "these", "they", "theyd", "theyll", "theyre", "theyve", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasnt", "we", "wed", "well", "were", "weve", "were", "werent", "what", "whats", "when", "whens", "where", "wheres", "which", "while", "who", "whos", "whom", "why", "whys", "with", "wont", "would", "wouldnt", "you", "youd", "youll", "youre", "youve", "your", "yours", "yourself", "yourselves")

vocab = create_vocabulary(it_train, stopwords = stop_words)

pruned_vocab = prune_vocabulary(vocabulary = vocab, 
                                term_count_min = 10, 
                                doc_proportion_max = 0.5, 
                                doc_proportion_min = 0.001)

vectorizer = vocab_vectorizer(pruned_vocab)

dtm_final_non_tfidf = create_dtm(it_train, vectorizer)

tfidf = TfIdf$new()
dtm_final = fit_transform(dtm_final_non_tfidf, tfidf)

colnames_dtm_final = rownames(t(dtm_final))

setwd("C:/Users/andre/GoogleDrive/MLPSProject/")

write(colnames_dtm_final, file = "Data/colnames_train_term_matrix_pruned.txt")
writeMM(dtm_final, file = "Data/train_term_matrix_pruned.mtx")



  