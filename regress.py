import csv, copy
import numpy as np
import pickle
from sklearn import cross_validation
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import argparse

model = GradientBoostingRegressor(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0, loss='ls')
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000, analyzer='word', ngram_range=(1, 3),
        token_pattern=ur'\b\w+\b', min_df=1)

    #  CountVectorizer(
    #     stop_words='english', 
    #     max_features=10000, 
    #     analyzer='word',
    #     ngram_range=(1,3), 
    #     token_pattern=ur'\b\w+\b',
    #     min_df=1)

def load_reddit(filename, vectorizer, num_examples=-1):
    data_file = csv.reader(open(filename), delimiter=',', quotechar='"')
    shape = next(data_file)
    n_examples = int(shape[0])
    n_features = int(shape[1])

    if num_examples > 0:
        n_examples = min(n_examples, num_examples)

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

def test_model(num_examples=-1):
    print 'Loading data'
    data, target = load_reddit('reddit.csv', vectorizer, num_examples)
    n_examples = data.shape[0]
    print "Testing model"
    cv = cross_validation.ShuffleSplit(n_examples, n_iter=10, test_size=.8)
    results = cross_validation.cross_val_score(model, data, target, cv=cv, scoring='r2', n_jobs=4)
    print sum(results)/len(results)

def build_save_model(num_examples=-1):
    print "Building model"
    data, target = load_reddit('reddit.csv', vectorizer, num_examples=num_examples)
    model.fit(data, target)
    print "Saving model"
    with open('model.pickle', '+wb') as model_file, open('vectorizer.pickle', '+wb') as vector_file:
        try:
            pickle.dump(model, model_file)
            pickle.dump(vectorizer, vector_file)
        except pickle.PicklingError as e:
            print "Error pickling"

def load_model():
    try:
        with open('model.pickle') as model_file, open('vectorizer.pickle') as vector_file:
            try:
                model = pickle.load(model_file)
                vectorizer = pickle.load(vector_file)    
            except pickle.UnpicklingError as e:
                print "Error unpickling"          
    except IOError as e:
        pass

if __name__ == '__main__': # run from command line
    parser = argparse.ArgumentParser(description='Build or test a model using reddit.csv file.')
    parser.add_argument('action', choices=['test', 'build'], default='test', help="")
    parser.add_argument('--debug', dest='num_examples', action='store', nargs='?', const='1000', default=-1, help="Train on a subset of given examples")
    args = parser.parse_args()
    print args.num_examples
    if args.action == 'test':
        test_model(args.num_examples)
    elif args.action == 'build':
        build_save_model(args.num_examples)
else: # imported
    load_model()