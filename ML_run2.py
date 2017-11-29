#!/usr/bin/env python
import cPickle as pickle
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
import time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report


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

    from sklearn import feature_selection
    import tempfile
    from sklearn.externals.joblib import Memory
    from sklearn.model_selection import KFold
    from sklearn.pipeline import Pipeline
    cachedir = tempfile.mkdtemp()
    mem = Memory(cachedir=cachedir, verbose=1)
    cv = KFold(10)

    f_classif = mem.cache(feature_selection.f_classif)
    #f_classif = feature_selection.f_classif
    anova = feature_selection.SelectPercentile(f_classif)

    print('ExtraTree Anova percentile starts')
    from sklearn.ensemble import ExtraTreesClassifier
    extratee = ExtraTreesClassifier(n_estimators=50)
    pipe = Pipeline([('anova', anova), ('decision', extratee)])
    clf_per = GridSearchCV(pipe, {'anova__percentile': range(5, 101, 5)}, cv=cv,
                           refit=True,
                           verbose=3)
    clf_per.fit(x_data, y_data)

    print('Best: %f using %s' % (clf_per.best_score_, clf_per.best_params_))


    print("KNN train starts")
    clf = KNeighborsClassifier()
    clf.fit(x_data, y_data)
    # Train KNN classifier
    # X_train, X_test, y_train, y_test = train_test_split(x_data, y_data,test_size=0.10,random_state=42)
    #clf.fit(x_data, y_data)
    #grid_clf.fit(x_data, y_data)

    #print('DecisionTreeClassifier with ROSE', clf.score(X_test, y_test))
    #print(len(x_data))
    #print(clf.get_params())
    #print(clf)
    print("KNN train FINISHED")

    #joblib.dump(clf, model_file)
    #with open(model_file, 'wb') as outfile:
    #    pickle.dump(clf, outfile, pickle.HIGHEST_PROTOCOL)
    #    outfile.close()


    # bring whole data to produce prediction
    xpath = '/home/junlinux/Desktop/CSCI544_Last/hw7/data/word_vector' \
            '/ROSE_wordlen_appear_freq_matchW-smoothed_ALL'
    x_data = load_wordvec(xpath)
    # min max scaling and normalization
    min_max_scaler = preprocessing.MinMaxScaler()
    x_data = min_max_scaler.fit_transform(x_data)
    x_data = preprocessing.normalize(x_data, norm='l2')

    print('Whole data loaded')

    print('Prediction file will be generated')
    predic_with_estimator(clf_per, x_data)
    print('File geneartion is done')


    """
    # GridCV test part
    # Split the dataset in two equal parts
    X_train, X_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.9, random_state=0)

    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    scores = ['precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                           scoring='%s_macro' % score)
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()
    """


def predic_with_estimator(clf, x_data):
    x_data_anova = clf.best_estimator_.named_steps['anova'].transform(x_data)
    pred = clf.best_estimator_.named_steps['decision'].predict(x_data_anova)
    #pred = clf.predict(x_data)
    print('pred', len(pred))
    with open('/home/junlinux/Desktop/CSCI544_Last/hw7/data/my.answers_ExT_test2', 'wb') as rd:
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
             '/ROSE_wordlen_appear_freq_matchW-smoothed'
    y_path = '/home/junlinux/Desktop/CSCI544_Last/hw7/data/dev.answers'
    model_file = 'data/model/model_test'
    train(x_path, y_path, 'g', model_file)

    #load_and_predict(model_file, x_path, y_path)
    #print("--- %s seconds ---" % (time.time() - start_time))


if __name__=='__main__':
    main()

