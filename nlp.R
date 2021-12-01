library(tidyverse)
library(tm)
library(SnowballC)
library(naivebayes)
library(randomForest)
library(caret)
library(caTools)


set.seed(2)
base_path = '/home/user/base_folder
setwd(vase_path')

# import and inspect data -------------------------------------------------------------

data <- read.csv2('data/labelled_documents.csv',sep = ';') %>% 
  select(facit,txt) %>% 
  rename(text=txt) %>% 
  mutate(facit = as.factor(facit)) %>% 
  distinct()

glimpse(data)


# Per group occurances
data %>% 
  group_by(facit) %>% 
  count() %>% 
  arrange(desc(n))

# Prepare text field -----------------------------------------------------------

fi_stopwords <- stopwords::stopwords(language = "fi")
swe_stopwords <- stopwords::stopwords(language = "sv")
eng_stopwords <- stopwords::stopwords()

cleaned_data <- data %>% 
  mutate(text = tolower(text),
         text = text %>% 
           str_remove_all('[[:punct:]]'))

post_stopfi <- tm::removeWords(cleaned_data$text,fi_stopwords)
post_stopswe <- tm::removeWords(post_stopfi,swe_stopwords)
post_stopeng <- tm::removeWords(post_stopswe,eng_stopwords)
cleaned_data$text <- post_stopeng

glimpse(cleaned_data)


# Create corpus -----------------------------------------------------------

corpus = Corpus(VectorSource(cleaned_data$text))

# Create word listings -----------------------------------------------------

frequencies = DocumentTermMatrix(corpus)

sparse = removeSparseTerms(frequencies, 0.99)

# Import data back into dataframe -----------------------------------------

tSparse = as_tibble(as.matrix(sparse),.name_repair = "universal")

colnames(tSparse) = make.names(colnames(tSparse))
tSparse$facit = data$facit

# Separate into train and test --------------------------------------------

split <- caTools::sample.split(tSparse$facit, SplitRatio = 0.7)
trainSparse <- subset(tSparse, split == TRUE)
testSparse <- subset(tSparse, split == FALSE)

# Separate x from y-----------------------------------------------------

X_train <- trainSparse %>% 
  select(-facit)

y_train <- trainSparse$facit

X_test <- testSparse %>% 
  select(-facit)

y_test <- testSparse$facit

# Train model -------------------------------------------------------------

mnb <- naivebayes::multinomial_naive_bayes(x = X_train,
                                           y = y_train,
                                           laplace = 1)

#Model info
summary(mnb)
coef(mnb)


# Check predictions -------------------------------------------------------

X_matrix <- as.matrix(X_test)
predictionsNB <- predict(mnb,
                         newdata = X_matrix,
                         type = "class")

confusionMatrix(predictionsNB, y_test)

