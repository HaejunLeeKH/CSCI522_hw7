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
from sklearn import feature_selection
import tempfile
from sklearn.externals.joblib import Memory
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier


# load vector data of sentence. Pre-made
def load_wordvec(vec_path):
    with open(vec_path, 'rb') as infile:
        result = pickle.load(infile)
    return result


# load class label data
def load_target(target_path):
    with open(target_path, 'rb') as infile:
        return [x.strip() for x in infile]


def train_ExtraTree(xpath, xpath_all, ypath, eval_path, model_file):

    # load pre-made vector data
    x_data = load_wordvec(xpath)
    y_data = load_target(ypath)

    # min max scaling and normalization
    min_max_scaler = preprocessing.MinMaxScaler()
    x_data = min_max_scaler.fit_transform(x_data)
    x_data = preprocessing.normalize(x_data, norm='l2')

    # create pipeline and setup GridSearchCV
    cachedir = tempfile.mkdtemp()
    mem = Memory(cachedir=cachedir, verbose=1)
    cv = KFold(10, random_state=43)
    f_classif = mem.cache(feature_selection.f_classif)
    # feature selection for pipeline
    anova = feature_selection.SelectPercentile(f_classif)
    #nn = MLPClassifier(hidden_layer_sizes=1500, max_iter=700,
                       #learning_rate='constant', batch_size=1000)
    knn = KNeighborsClassifier()
    # this parameter is found test stage.
    #an = {'anova__percentile': [85]}
    an = {'anova__percentile': range(5, 101, 5),
          'weights':['uniform', 'distance'],
          'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],
          'DistanceMetric':['euclidean', 'manhattan', 'chebyshev', 'minkowski',
                            'wminkowski', 'seuclidean', 'mahalanobis']}
    cv = KFold(10, random_state=43)
    nn_pipe = Pipeline([('anova', anova), ('knn', knn)])
    clf_per = GridSearchCV(nn_pipe, an, cv=cv,
                           refit=True,
                           verbose=3)
    # {'anova__percentile': range(5, 101, 5)}

    print('Neural Network Anova percentile starts')

    clf_per.fit(x_data, y_data)

    # show best result with parameter, in this case, anova percentile
    print('Best: %f using %s' % (clf_per.best_score_, clf_per.best_params_))

    # save model for future use
    joblib.dump(clf_per, model_file)
    with open(model_file, 'wb') as outfile:
        pickle.dump(clf_per, outfile, pickle.HIGHEST_PROTOCOL)
        outfile.close()


    # bring whole data to produce prediction
    x_data = load_wordvec(xpath_all)
    # min max scaling and normalization
    min_max_scaler = preprocessing.MinMaxScaler()
    x_data = min_max_scaler.fit_transform(x_data)
    x_data = preprocessing.normalize(x_data, norm='l2')

    print('Whole data loaded')
    print('Prediction file will be generated')
    # Do prediction with generated model and save it as file
    predic_with_estimator(clf_per, x_data, eval_path)
    print('File geneartion is done')


def predic_with_estimator(clf, x_data, eval_path):

    # follow steps of best estimator in classifier object
    x_data_anova = clf.best_estimator_.named_steps['anova'].transform(x_data)
    pred = clf.best_estimator_.named_steps['knn'].predict(x_data_anova)
    #pred = clf.predict(x_data)

    # check the number of predictions model made
    print('pred', len(pred))
    with open(eval_path, 'wb') as rd:
        for p in pred:
            rd.write(str(p)+'\n')


# prediction with EINTRE data doesn't require target data
def load_and_predict(model_path, xpath, eval_path):

    # load model data
    with open(model_path, 'rb') as input:
        model = pickle.load(input)
    print(model)
    print(model.get_params())

    # load data to predict. Need to get ENTIRE data set path
    x_data = load_wordvec(xpath)

    # data size check
    print('x_data', len(x_data))

    # min max scaling and normalization
    min_max_scaler = preprocessing.MinMaxScaler()
    x_data = min_max_scaler.fit_transform(x_data)
    x_data = preprocessing.normalize(x_data, norm='l2')

    print('getting scores')
    # make prediction and save it into file
    predic_with_estimator(model, x_data, eval_path)


def main():
    # if train and predict from vector file,
    # use train function.
    # if load from pre-trained file,
    # use load_and_predict file

    # evaluation file will be generated at eval_path1 and eval_path2
    # stdout for this program was too fast, so output redirection to
    # check and evaluation file didn't work
    # Thus, you have to save predictio to file first,
    # and bring it to check and compare-evaluation file
    # you can do it by
    # cat data/my.answers_Ext_anova_2 | ./check | ./compare-with-human-evaluation
    # where cat [evaluation file path]

    start_time = time.time()
    x_path = 'data_sub' \
             '/ROSE_wordlen_appear_freq_smoothed'
    #x_path = 'data_sub' \
         #'/ROSE_wordlen_appear_freq_smoothed_LASTHALF'

    xpath_all = 'data_sub' \
                '/ROSE_wordlen_appear_freq_smoothed_ALL'
    #xpath_all = 'data_sub' \
     #'/ROSE_wordlen_appear_freq_smoothed'

    y_path = 'data_sub/dev.answers'
    #y_path = 'data_sub/nn_last_answer'

    #eval_path = 'data_sub/my.answers_Ext_anova_1_LAST'
    eval_path = 'data_sub/my.answers_NN_anova_1'
    eval_path2 = 'data_sub/my.answers_NN_anova_2'

    model_path1 = 'data_sub/model_nn_1'
    model_path2 = 'data_sub/model_Ext_2'
    train_ExtraTree(x_path, xpath_all, y_path, eval_path, model_path1)

    #load_and_predict(model_path1, xpath_all, eval_path2)
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__=='__main__':
    main()

