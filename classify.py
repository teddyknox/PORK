import csv, copy
import pylab as pl
import numpy as np
from sklearn import datasets, linear_model, cross_validation

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
MAX_SIZE = 100
total_labs = []
total_bods = []

# vectorizer = TfidfVectorizer(stop_words='english', max_features=10000, analyzer='word', ngram_range=(1, 3), token_pattern=ur'\b\w+\b', min_df=1)
vectorizer = CountVectorizer(stop_words='english', max_features=10000, analyzer='word', ngram_range=(1,3), token_pattern=ur'\b\w+\b', min_df=1)


dataset_size = 132307

# cross validation by array indicies
rs = cross_validation.ShuffleSplit(dataset_size / 10, n_iter=2, test_size=.9)
train_indicies = []
test_indicies = []

for train_index, test_index in rs:
  # for every n_iter append the train_index indicies
  train_indicies.append(train_index)

  # for every n_iter append the test_index indicies
  test_indicies.append(test_index)



print "Building Dictionaries"
with open('reddit.csv', 'rb') as csvfile:
  filereader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
  i = 0
  for row in filereader:
    i += 1
    total_labs.append(int(row["number_of_upvotes"]) - int(row["number_of_downvotes"]))
    total_bods.append(copy.deepcopy(row["title"]))

correct = 0.0
total = 0.0

# Run for n times cross validatation
for i in range(0, len(train_indicies)):
  # Train indices
  train_row = train_indicies[i]
  # Test indices
  test_row = test_indicies[i]
  train_labs = []
  train_bods = []
  test_bods = []
  test_labs = []
  for i in train_row:
    train_labs.append(total_labs[i])
    train_bods.append(total_bods[i])
  for i in test_row:
    test_labs.append(total_labs[i])
    test_bods.append(total_bods[i])

  # Fit training/testing vectors
  print "Vectorizing"
  vectorizer.fit(total_bods)
  train_vecs = vectorizer.transform(train_bods)
  train_vecs = train_vecs.toarray()
  test_vecs = vectorizer.transform(train_bods)
  test_vecs = test_vecs.toarray()

  m = linear_model.LinearRegression()
  print "Fitting"
  m.fit(train_vecs, train_labs)


  print results
