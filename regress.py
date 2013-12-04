import csv, copy
import numpy as np
import pickle
from sklearn import cross_validation
from sklearn.ensemble import GradientBoostingRegressor, LinearRegression, LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import argparse

class Model(object):
    def __init__(self, filename=None, num_examples=None, force_train=False):
        self.filename = filename
        self.num_examples = num_examples
        
        # short-circuit conditional
        if (not force_train and not self.unpickle()) or force_train: # if we don't want to retrain and we have nothing to load
            self.reg = LogisticRegression()
            # self.reg = GradientBoostingRegressor(                    # or we do want to retrain
            #     n_estimators=100,
            #     learning_rate=1.0,
            #     max_depth=1,
            #     random_state=0, 
            #     loss='ls')
            # self.vectorizer = TfidfVectorizer(
            #     stop_words='english', 
            #     max_features=10000, 
            #     analyzer='word', 
            #     ngram_range=(1, 3),
            #     token_pattern=ur'\b\w+\b', 
            #     min_df=1)
            self.vectorizer = CountVectorizer(
                # strip_accents='unicode', 
                # max_features=10000,
                # analyzer='word', 
                # token_pattern=ur'\b\w+\b', 
                # lowercase=True, 
                ngram_range=(1,1)
            )
            if filename:
                self.train(filename, num_examples=num_examples)
                self.trained = True
                self.save()
            else:
                self.trained = False

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
            print "Saving model"
            with open('model.pickle', 'wb') as model_file, open('vectorizer.pickle', 'wb') as vector_file:
                try:
                    pickle.dump(self.reg, model_file)
                    pickle.dump(self.vectorizer, vector_file)
                except pickle.PicklingError as e:
                    print "Error pickling"
        else:
            print "Model not trained, so why save?."

    def train(self, filename=None, num_examples=None):
        print "Building model"
        fn = filename or self.filename
        ne = num_examples or self.num_examples        
        data, target = self.load_reddit_csv(filename=fn, num_examples=ne)
        self.reg.fit(data, target)
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
        cv = cross_validation.ShuffleSplit(n_examples, n_iter=10, test_size=.8)
        results = cross_validation.cross_val_score(self.reg, data, target, cv=cv, scoring='r2', n_jobs=4)
        print sum(results)/len(results)

    def load_reddit_csv(self, filename, num_examples=None):
        data_file = csv.reader(open(filename), delimiter=',', quotechar='"')
        shape = next(data_file)
        n_examples = int(shape[0])
        n_features = int(shape[1])

        if num_examples:
            n_examples = min(n_examples, int(num_examples))

        # data = np.empty( (n_examples, n_features) )
        titles = []
        target = np.empty((n_examples,), dtype=np.int32)
        feature_names = next(data_file)
        np.array(feature_names)
        for i, d in enumerate(data_file):
            if i == n_examples:
                break
            titles.append(d[3]) # the tenth item is the 'score'
            target[i] = d[10]
        data = self.vectorizer.fit_transform(titles).toarray()
        return data, target

if __name__ == '__main__': # run from command line
    parser = argparse.ArgumentParser(description='Build or test a model using reddit.csv file.')
    parser.add_argument('action', choices=['test', 'build'], default='test', help="")
    parser.add_argument('--debug', dest='num_examples', action='store', nargs='?', const='1000', help="Train on a subset of given examples")
    args = parser.parse_args()

    # initialize model
    if args.action == 'test':
        model = Model('test.csv', num_examples=args.num_examples)
        model.train_test()
    elif args.action == 'build':
        model = Model('test.csv', num_examples=args.num_examples, force_train=True)