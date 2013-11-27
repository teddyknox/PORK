import csv
import pylab as pl
import numpy as np
from sklearn import datasets, linear_model, cross_validation

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
MAX_FEATURES = 100
train_labs = []
train_bods = []

# vectorizer = TfidfVectorizer(stop_words='english', max_features=10000, analyzer='word', ngram_range=(1, 3), token_pattern=ur'\b\w+\b', min_df=1)
vectorizer = CountVectorizer(stop_words='english', max_features=10000, analyzer='word', ngram_range=(1,3), token_pattern=ur'\b\w+\b', min_df=1)

i = 0

dataset_size = 132307

# cross validation by array indicies
rs = cross_validation.ShuffleSplit(dataset_size, n_iter=2, test_size=.9)
train_indicies = []
test_indicies = []

for train_index, test_index in rs:
  # for every n_iter append the train_index indicies
  train_indicies.append(train_index)

  # for every n_iter append the test_index indicies
  test_indicies.append(test_index)


for row in train_indicies:
  print row

for row in test_indicies:
  print row


print "Building Dictionaries"
with open('reddit.csv', 'rb') as csvfile:
  spamreader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
  row_count = sum(1 for row in spamreader)
  print row_count
  for row in spamreader:
    i+=1
    if i < MAX_FEATURES:
      train_labs.append(int(row["number_of_upvotes"]) - int(row["number_of_downvotes"]))
      train_bods.append(row["title"])
      # print i, row["number_of_upvotes"], row["number_of_downvotes"], int(row["number_of_upvotes"]) - int(row["number_of_downvotes"])
      # print ', '.join(row)
      # 

print "Vectorizing"
vectorizer.fit(train_bods)
train_vecs = vectorizer.transform(train_bods)
train_vecs = train_vecs.toarray()
m = linear_model.LinearRegression()
print "Fitting"
m.fit(train_vecs, train_labs)

