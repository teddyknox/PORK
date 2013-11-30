import csv, copy
import pylab as pl
import numpy as np
from sklearn import datasets, linear_model, cross_validation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# set to -1 to use all examples
TEST_SIZE = 1000

# util anon funcs
average = lambda X: sum(X)/len(X)

# load data
def load_reddit(filename, vectorizer=None, test_size=-1):
    data_file = csv.reader(open(filename), delimiter=',', quotechar='"')
    shape = next(data_file)
    n = n_examples = int(shape[0])
    n_features = int(shape[1])

    if test_size > 0 and test_size < n_examples:
        n = test_size

    # data = np.empty( (n_examples, n_features) )
    titles = []
    target = np.empty((n_examples,), dtype=np.int32)
    feature_names = next(data_file)
    np.array(feature_names)
    for i, d in enumerate(data_file):
        if i == n:
            break
        titles.append(d[10])
        target[i] = d[10]
    if not vectorizer:
        vec_params = {
            'stop_words': 'english', 
            'max_features': 10000, 
            'analyzer': 'word',
            'ngram_range': (1,3), 
            'token_pattern': ur'\b\w+\b',
            'min_df': 1
        }
        vectorizer = CountVectorizer(**vec_params)
    data = vectorizer.fit(titles).transform(titles).toarray()
    return data, target

print 'Loading data'
# vec = TfidfVectorizer(stop_words='english', max_features=10000, analyzer='word', ngram_range=(1, 3), token_pattern=ur'\b\w+\b', min_df=1)
data, target = load_reddit('reddit.csv', test_size=TEST_SIZE)

print "Testing model"
reg = linear_model.LinearRegression()
cv = cross_validation.ShuffleSplit(TEST_SIZE, n_iter=10, test_size=.8)
results = cross_validation.cross_val_score(reg, data, np.array(target), cv=cv, scoring='r2', n_jobs=4)
print average(results)

# putting this down here for now

# cross validation by array indicies
# rs = cross_validation.ShuffleSplit(dataset_size / 10, n_iter=2, test_size=.9)
# train_indicies = []
# test_indicies = []

# for train_index, test_index in rs:
#   # for every n_iter append the train_index indicies
#   train_indicies.append(train_index)

#   # for every n_iter append the test_index indicies
#   test_indicies.append(test_index)