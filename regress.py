import csv, copy
import numpy as np
from sklearn import cross_validation
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# set debug param to False to disable
TEST_SIZE = 1000

model = GradientBoostingRegressor(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0, loss='ls')
vectorizer = TfidfVectorizer(
        stop_words='english', 
        max_features=10000, 
        analyzer='word',
        ngram_range=(1, 3),
        token_pattern=ur'\b\w+\b',
        min_df=1)
    #  CountVectorizer(
    #     stop_words='english', 
    #     max_features=10000, 
    #     analyzer='word',
    #     ngram_range=(1,3), 
    #     token_pattern=ur'\b\w+\b',
    #     min_df=1)

# load data
def load_reddit(filename, debug=False):
    data_file = csv.reader(open(filename), delimiter=',', quotechar='"')
    shape = next(data_file)
    n_examples = int(shape[0])
    n_features = int(shape[1])

    if debug:
        n_examples = min(n_examples, TEST_SIZE)

    # data = np.empty( (n_examples, n_features) )
    titles = []
    target = np.empty((n_examples,), dtype=np.int32)
    feature_names = next(data_file)
    np.array(feature_names)
    for i, d in enumerate(data_file):
        if i == n_examples:
            break
        titles.append(d[10])
        target[i] = d[10]
    data = vectorizer.fit_transform(titles).toarray()
    return data, target

def test_model(debug=False):
    print 'Loading data'
    data, target = load_reddit('reddit.csv', debug)
    n_examples = data.shape[0]
    print "Testing model"
    cv = cross_validation.ShuffleSplit(n_examples, n_iter=10, test_size=.8)
    results = cross_validation.cross_val_score(model, data, target, cv=cv, scoring='r2', n_jobs=4)
    print sum(results)/len(results)

if __name__ == '__main__':
    test_model(debug=True)