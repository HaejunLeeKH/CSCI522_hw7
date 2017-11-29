#!/usr/bin/env python
import cPickle as pickle
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
import time


# load vector data of sentence. Pre-made
def load_wordvec(vec_path):
    with open(vec_path, 'rb') as infile:
        result = pickle.load(infile)
    return result


# load class label data
def load_target(target_path):
    with open(target_path, 'rb') as infile:
        return [x.strip() for x in infile]


def train(xpath, ypath, method, model_file='data/model/model_test'):

    # load pre-made vector data
    x_data = load_wordvec(xpath)

    y_data = load_target(ypath)

    # min max scaling and normalization
    min_max_scaler = preprocessing.MinMaxScaler()
    x_data = min_max_scaler.fit_transform(x_data)
    x_data = preprocessing.normalize(x_data, norm='l2')

    clf = KNeighborsClassifier()
    scores = cross_val_score(clf, x_data, y_data, cv=3)
    print('CV3', scores)
    #scores = cross_val_score(clf, x_data, y_data, cv=5)
    #print('CV5', scores)

    clf.fit(x_data, y_data)
    #print(len(x_data))
    #print(clf.get_params())
    #print(clf)


    #joblib.dump(clf, model_file)
    with open(model_file, 'wb') as outfile:
        pickle.dump(clf, outfile, pickle.HIGHEST_PROTOCOL)
        outfile.close()

    # predic_with_estimator(clf, x_data)


def predic_with_estimator(clf, x_data):

    pred = clf.predict(x_data)
    print('pred', len(pred))
    with open('/home/junlinux/Desktop/CSCI544_Last/hw7/data/my.answers', 'wb') as rd:
        for p in pred:
            rd.write(str(p)+'\n')


def load_and_predict(model_path, xpath, ypath):

    with open(model_path, 'rb') as input:
        model = pickle.load(input)
    print(model)
    print(model.get_params())
    x_data = load_wordvec(xpath)
    y_data = load_target(ypath)
    print('x_data', len(x_data))

    # min max scaling and normalization
    min_max_scaler = preprocessing.MinMaxScaler()
    x_data = min_max_scaler.fit_transform(x_data)
    x_data = preprocessing.normalize(x_data, norm='l2')

    print('getting scores')

    print(model.score(x_data, y_data))

def main():
    start_time = time.time()
    x_path = '/home/junlinux/Desktop/CSCI544_Last/hw7/data/word_vector' \
             '/ROSE_wordlen'
    y_path = '/home/junlinux/Desktop/CSCI544_Last/hw7/data/dev.answers'
    model_file = 'data/model/model_test'
    train(x_path, y_path, 'g', model_file)

    #load_and_predict(model_file, x_path, y_path)
    #print("--- %s seconds ---" % (time.time() - start_time))


if __name__=='__main__':
    main()

