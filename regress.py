import csv, copy
import numpy as np
import pickle
import random
import time
from sklearn import cross_validation
from sklearn.svm import SVR
from sklearn.feature_selection import RFECV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.linear_model import LogisticRegression
import argparse

class Model(object):
    def __init__(self, filename=None, num_examples=None, force_train=False):
        self.filename = filename
        self.num_examples = num_examples
        # short-circuit conditional
        if force_train or not self.unpickle(): # if we don't want to retrain and we have nothing to load
            self.reg = SVR(kernel='linear', C=1e3, gamma=0.1)
            # self.reg = LogisticRegression(penalty='l2')
            # self.vectorizer = CountVectorizer(ngram_range=(1,3))
            self.vectorizer = TfidfVectorizer(norm='l2', smooth_idf=True, sublinear_tf=True, use_idf=True)
            if filename:
                self.train(filename, num_examples=num_examples)
                self.trained = True
                self.save()
            else:
                self.trained = False
        else:
            print "Already trained."

    def unpickle(self):
        try:
            with open('model.pickle') as model_file, open('vectorizer.pickle') as vector_file:
                try:
                    # self.model = pickle.load(model_file)
                    self.reg = pickle.load(model_file)
                    self.vectorizer = pickle.load(vector_file)
                    return True
                except pickle.UnpicklingError as e:
                    print "Error unpickling"
                    return False
        except IOError as e:
            print "No pickle files found"
            return False

    def save(self):
        if self.trained:
            start = time.time()
            print "Saving model"
            with open('model.pickle', 'wb') as model_file, open('vectorizer.pickle', 'wb') as vector_file:
                try:
                    pickle.dump(self.reg, model_file)
                    pickle.dump(self.vectorizer, vector_file)
                except pickle.PicklingError as e:
                    print "Error pickling"
            print time.time() - start, "seconds to save"
        else:
            print "Model not trained, so why save?"

    def train(self, filename=None, num_examples=None):
        print "Building model"
        start = time.time()
        fn = filename or self.filename
        ne = num_examples or self.num_examples        
        print "Loading data"
        data, target = self.load_reddit_csv(filename=fn, num_examples=ne)
        print "Training model"
        print time.time() - start, "seconds to vectorize"
        start = time.time()
        print "Training model"
        self.reg.fit(data, target)
        print time.time() - start, "seconds to train"
        self.trained = True

    def predict(self, title):
        ''' Takes as input a title string and returns the predicted score value. '''
        X = self.vectorizer.transform([title]).toarray()
        y = self.reg.predict(X)
        return y[0]

    def train_test(self, filename=None, num_examples=None):
        ''' Trains and academically tests the regressor. Does not trip the trained flag.'''
        fn = filename or self.filename
        ne = num_examples or self.num_examples 
        print 'Loading data'
        data, target = self.load_reddit_csv(fn, ne)
        n_examples = data.shape[0]
        print "Testing model"
        cv = cross_validation.ShuffleSplit(n_examples, n_iter=4, test_size=.8)
        results = cross_validation.cross_val_score(self.reg, data, target, cv=cv, scoring='r2', n_jobs=4)
        print results

    def load_reddit_csv(self, filename, num_examples=None, split=0.8):
        data_file = csv.reader(open(filename), delimiter=',', quotechar='"')
        shape = next(data_file)
        n_examples = int(shape[0])
        n_features = int(shape[1])
        if num_examples:
            n_examples = min(n_examples, num_examples)

        feature_names = np.array(next(data_file))

        # data = np.empty( (n_examples, n_features) )
        target_list = list(array_generator(data_file, 10, n_examples))
        titles = array_generator(data_file, 3, n_examples)

        target = np.array(target_list, dtype=np.int32)
        data = self.vectorizer.fit_transform(titles).toarray()
        return data, target

def calc_stats(filename, num_examples=None):
    data_file = csv.reader(open(filename), delimiter=',', quotechar='"')
    shape = next(data_file)
    names = next(data_file)
    n_examples = int(shape[0])
    n_features = int(shape[1])
    if num_examples:
        n_examples = min(n_examples, int(num_examples))
    i = 0
    avg_length = 0
    avg_score = 0
    avg_scores = {}
    avg_counts = {}
    max_score = float("-inf")
    min_score = float("inf")
    for i, n in enumerate(data_file):
        if i >= n_examples:
            break
        score = int(n[10])
        length = len(n[3].split(" "))
        if not length in avg_scores:
            avg_scores[length] = 0
        if not length in avg_counts:
            avg_counts[length] = 0
        avg_scores[length] += int(n[10])
        avg_counts[length] += 1
        avg_length += length
        avg_score += score
        if score > max_score:
            max_score = score
        if score < min_score:
            min_score = score
    print "Average Title Length:\t", float(avg_length) / n_examples
    print "Average Score:\t\t", float(avg_score) / n_examples
    print "Maximum Score:\t\t", float(max_score)
    print "Minimum Score:\t\t", float(min_score)

    for k in avg_scores.keys():
        val = avg_scores[k]
        print "Average",k,"word title score:", float(val) / avg_counts[k]

def array_generator(iterator, index, n):
    i = 0
    while i < n:
        yield next(iterator)[index] # title
        i += 1

if __name__ == '__main__': # run from command line
    parser = argparse.ArgumentParser(description='Build or test a model using reddit.csv file.')
    parser.add_argument('--test', action='store_true', help="Whether to test")
    parser.add_argument('--debug', dest='num_examples', type=int, action='store', nargs='?', const='1000', help="Train on a subset of given examples")
    parser.add_argument('--force', dest='force_train', action='store_true', help="Force retrain dataset")    
    parser.add_argument('--stats', dest='calc_stats', action='store_true', help="Force retrain dataset")    
    args = parser.parse_args()

    if args.calc_stats:
        calc_stats('reddit.csv', num_examples=args.num_examples)
    else:
        # initialize model
        model = Model('reddit.csv', num_examples=args.num_examples, force_train=args.force_train)    
        if args.test:
            model.train_test()