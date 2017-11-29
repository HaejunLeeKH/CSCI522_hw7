# -*- coding: utf-8 -*-

from nltk.tokenize import word_tokenize
import string
import gensim
import numpy as np
import cPickle as pickle
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

mean_vec = 1
mean_tf = 2
mean_tf_whole = 3

path = 'data/data_test'  # whole sentence data
test_30_sentence = 'data/stem_testdata'  # only 30 sentence for function writing
v_path = "data/glove.6B/glove.6B.50d.txt"


# read data as three sentences: h1, h2, reference
def sentences(filepath):
    with open(filepath) as f:
        # with codecs.open(opts.input, 'r', encoding='utf-8') as f:
        for pair in f:
            yield [sentence.strip().decode('utf-8') for sentence in pair.split(' ||| ')]


# read h1, h2, ref as word list
def sentences2():
    with open(path) as f:
        # with codecs.open(opts.input, 'r', encoding='utf-8') as f:
        for pair in f:
            yield [sentence.strip().split() for sentence in pair.split(' ||| ')]


# class for generating mean vector of every words in sentence
class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        # load word vector file. It's downloaded from glove page
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(word2vec.itervalues().next())

    # this is use for pipelining with scikit learn
    def fit(self, X, y):
        return self

    def transform(self, X):

        # X: list containing 3 sentence, h1, h2, ref
        # for every words, if they are found in word vector, use that vector
        # if don't, then add zero vector instead
        result = np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

        return result


# class using Tf-Idf information
class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        # store word vec in object
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(word2vec.itervalues().next())

    # using TfidfVectorizer class from scikit learn
    # for this specific process, it didn't help much
    # probably requires larger data set
    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        # generate word vectors with weights
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])


# load word vector data downloaded from glove page
# they are used to transfrom sentence to word vectors
def word_vec_load(vpath="data/glove.6B/glove.6B.50d.txt"):
    with open(vpath, "rb") as lines:
        w2v = {line.split()[0]: np.array(map(float, line.split()[1:]))
               for line in lines}
        return w2v


# this function is for force Tf-Idf information is created with
# whole data set. It wasn't helpful much yet
def tftrain(filepath, glove_vec):
    tfvec = TfidfEmbeddingVectorizer(glove_vec)
    words = []
    for h1, h2, ref in sentences(filepath):
        words.extend([x.strip(string.punctuation) for x in h1.split()])
        words.extend([x.strip(string.punctuation) for x in h2.split()])
        words.extend([x.strip(string.punctuation) for x in ref.split()])
    tfvec.fit(words,[])
    return tfvec


def synonym(split):
    # find all synonyms and if any of them are in reference,
    # then use it instead
    h1 = []
    for hw in split[0]:
        if hw not in split[2]:
            # find all synonyms of word in hw
            syns = set([j.name() for i in wn.synsets(hw) for j in i.lemmas()])
            # if there is none, stop
            if len(syns)!=0:
                # if there is a match between synonyms and reference, use it
                for sword in syns:
                    if sword in split[2]:
                        h1.append(sword)
                        break
                    else:
                        h1.append(hw)
            else:
                h1.append(hw)
        else:
            h1.append(hw)
    h2 = []
    for hw in split[1]:
        if hw not in split[2]:
            syns = set([j.name() for i in wn.synsets(hw) for j in i.lemmas()])
            if len(syns)!=0:
                for sword in syns:
                    if sword in split[2]:
                        h2.append(sword)
                        break
                    else:
                        h2.append(hw)
            else:
                h2.append(hw)
        else:
            h2.append(hw)

    return [h1, h2, split[2]]


# this function is to differentiate matched words and non-
# matched words after synonym search, so that we can have more information
# Was not useful much yet
def synonym_divide(split):

    h1_in = []
    h1_out = []
    for hw in split[0]:
        if hw not in split[2]:
            syns = set([j.name() for i in wn.synsets(hw) for j in i.lemmas()])
            if len(syns)!=0:
                for sword in syns:
                    if sword in split[2]:
                        h1_in.append(sword)
                        break
                    else:
                        h1_out.append(hw)
            else:
                h1_out.append(hw)
        else:
            h1_in.append(hw)

    h2_in = []
    h2_out = []
    for hw in split[1]:
        if hw not in split[2]:
            syns = set([j.name() for i in wn.synsets(hw) for j in i.lemmas()])
            if len(syns)!=0:
                for sword in syns:
                    if sword in split[2]:
                        h2_in.append(sword)
                        break
                    else:
                        h2_out.append(hw)
            else:
                h2_out.append(hw)
        else:
            h2_in.append(hw)

    ref_in = []
    ref_out = []
    for r in split[2]:
        if r in h1_in+h2_in:
            ref_in.append(r)
        else:
            ref_out.append(r)

    return [h1_in, h1_out, h2_in, h2_out, ref_in, ref_out]


# this function is for dividing matched and non-matched words
# Was not useful much yet
def inter_out(split):

    h1_in = []
    h1_out = []
    for hw in split[0]:
        if hw not in split[2]:
            h1_out.append(hw)
        else:
            h1_in.append(hw)

    h2_in = []
    h2_out = []
    for hw in split[1]:
        if hw not in split[2]:
            h2_out.append(hw)
        else:
            h2_in.append(hw)

    ref_in = []
    ref_out = []
    for r in split[2]:
        if r in h1_in+h2_in:
            ref_in.append(r)
        else:
            ref_out.append(r)

    #return [h1_in, h1_out, h2_in, h2_out, ref_in, ref_out]
    return [h1_out, h2_out, ref_in, ref_out]


# generate word vector of sentence
def word_vec_gen(filepath, vecfile, vectype, vec_path):
    # filepath: data file path
    # vecfile: vector file path
    # vectype: which method will be used for vector generation

    glove_vec = word_vec_load(vecfile)
    vectorizer = None
    if vectype==1:
        # MeanVectorizer assign
        vectorizer = MeanEmbeddingVectorizer(glove_vec)
    elif vectype==2:
        # tf-idf, set one line as document for tf-idf
        vectorizer = TfidfEmbeddingVectorizer(glove_vec)
    elif vectype==3:
        # tf-idf, set whole data as document for tf-idf
        vectorizer = TfidfEmbeddingVectorizer(glove_vec)
        whole_tf = tftrain(filepath, glove_vec)

    all_mean = []
    for h1, h2, ref in sentences(filepath):
        split = [[x.strip(string.punctuation) for x in h1.split()],
                 [x.strip(string.punctuation) for x in h2.split()],
                 [x.strip(string.punctuation) for x in ref.split()]
                 ]
        # Here, we use inter_out or synonym_divide to have more vectors
        # Was not useful much yet
        # split = inter_out(split)

        # vectype 1 - average of all word vectors
        if vectype == 1:
            means = vectorizer.transform(split)
            temp = []

            for head in zip(means[0], means[1], means[2]):
                temp.extend(head)

            all_mean.append(temp)


        # vectype 2 - average + tf-idf by one line of data, each time
        # second parameter of fit fuction doesn't do anything with weight calculation
        # thus, we feed 1s
        elif vectype == 2:
            vectorizer.fit(split, [1])
            means = vectorizer.transform(split)
            all_mean.append(np.concatenate((means[0], means[1], means[2]), axis=0))

        # vectype 3 - average + tf-idf by whole data
        elif vectype == 3:
            means = whole_tf.transform(split)
            all_mean.append(np.concatenate((means[0], means[1], means[2]), axis=0))

    # save and load numpy array to and from file
    # to preserve the order, which is important to match with target
    # using pickle
    if len(all_mean) > 0:
        with open(vec_path, 'wb') as outfile:
            pickle.dump(all_mean, outfile, pickle.HIGHEST_PROTOCOL)
            print(vecfile, vectype, len(all_mean), 'vectors SUCCEED!')
    else:
        print(vecfile, vectype, 'failed to word vec generation.')
        print('vectype 1 - average, 2 - tf-idf by one line, 3 - tf-idf by whole data')


# load word vector of sentence
def load_wordvec(vec_path):
    with open(vec_path, 'rb') as infile:
        result = pickle.load(infile)
    return result


# load class label
def load_target(target_path):
    with open(target_path, 'rb') as infile:
        return [x.strip() for x in infile]


def main():
    # filepath: sentence data file path
    # vecfile: word vector file path pre-generated from other
    # vectype: compression methods. Average, avg+tf-idf one line, agg+tf-idf whole data
    # vec_path: vector file save path

    filepath =  '/home/junlinux/Desktop/CSCI544_Last/hw7/data/stem_testdata' # 'data/data_test'
    vecfile = '/home/junlinux/Desktop/CSCI544_Last/hw7/data/glove.6B/glove.6B.50d.txt'

    vec_files = ['/home/junlinux/Desktop/CSCI544_Last/hw7/data/glove.6B/glove.6B.50d.txt'
     ,'/home/junlinux/Desktop/CSCI544_Last/hw7/data/glove.6B/glove.6B.100d.txt'
     ,'/home/junlinux/Desktop/CSCI544_Last/hw7/data/glove.6B/glove.6B.200d.txt'
     ,'/home/junlinux/Desktop/CSCI544_Last/hw7/data/glove.6B/glove.6B.300d.txt'
     ,'/home/junlinux/Desktop/CSCI544_Last/hw7/data/glove.6B/glove.42B.300d.txt'
     ,'/home/junlinux/Desktop/CSCI544_Last/hw7/data/glove.6B/glove.840B.300d.txt'
                 ]
    # don't know why yet, relative file path having permission deny
    # so we're using absolute path for now
    vec_path = '/home/junlinux/Desktop/CSCI544_Last/hw7/data/word_vector/'


    # Here, we can choose type of vectorization
    # there are 6 word vector file downloaded from glove
    """
    vectype = 1
    for v in vec_files:
        start_time = time.time()
        name = v.split('/')[-1][:-4] + '_vec'
        print(name, 'vectorization in process')
        word_vec_gen(filepath, v, vectype, vec_path+name)
        print("--- %s seconds ---" % (time.time() - start_time))

    vectype = 2
    for v in vec_files:
        start_time = time.time()
        name = v.split('/')[-1][:-4] + '_vec_OnelineTF'
        print(name, 'vectorization in process')
        word_vec_gen(filepath, v, vectype, vec_path + name)
        print("--- %s seconds ---" % (time.time() - start_time))

    vectype = 3
    for v in vec_files:
        start_time = time.time()
        name = v.split('/')[-1][:-4] + '_vec_WholeDataTF'
        print(name, 'vectorization in process')
        word_vec_gen(filepath, v, vectype, vec_path + name)
        print("--- %s seconds ---" % (time.time() - start_time))
    """

    # from here, will earase.

    filepath = '/home/junlinux/Desktop/CSCI544_Last/hw7/data/data_test' # 'data/stem_testdata'
    #filepath = '/home/junlinux/Desktop/CSCI544_Last/hw7/data/hyp1-hyp2-ref'
    vectype = 1
    start_time = time.time()
    name = vecfile.split('/')[-1][:-4] + '_vec_diffOrder'
    #print(name, 'vectorization in process')
    #word_vec_gen(filepath, vecfile, vectype, vec_path + name)
    #print("--- %s seconds ---" % (time.time() - start_time))

    filepath = '/home/junlinux/Desktop/CSCI544_Last/hw7/data/data_test'  # 'data/stem_testdata'
    vectype = 2
    start_time = time.time()
    name = vecfile.split('/')[-1][:-4] + '_vec_OnelineTF'
    #print(name, 'vectorization in process')
    #word_vec_gen(filepath, vecfile, vectype, vec_path + name)
    #print("--- %s seconds ---" % (time.time() - start_time))

    filepath = '/home/junlinux/Desktop/CSCI544_Last/hw7/data/data_test'  # 'data/stem_testdata'
    vectype = 3
    start_time = time.time()
    name = vecfile.split('/')[-1][:-4] + '_vec_WholeDataTF'
    #print(name, 'vectorization in process')
    #word_vec_gen(filepath, vecfile, vectype, vec_path + name)
    #print("--- %s seconds ---" % (time.time() - start_time))


    vec_path = 'data/word_vector/glove.6B.50d_vec_diffOrder'
    wvec = load_wordvec(vec_path)
    target_path = 'data/dev.answers'
    answer = load_target(target_path)


    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import BernoulliNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.tree import ExtraTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import NuSVC
    from sklearn.multiclass import OneVsOneClassifier
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.svm import LinearSVC

    clf1 = KNeighborsClassifier()
    clf2 = DecisionTreeClassifier()
    clf3 = ExtraTreeClassifier()
    clf4 = MLPClassifier()
    clf5nu  =NuSVC()
    clf6lin = LinearSVC()
    # 'sag', 'saga' and 'lbfgs' ’



    print("Training Starts")
    X_train, X_test, y_train, y_test = train_test_split(wvec, answer,test_size=0.10,random_state=42)
    #clf1.fit(X_train, y_train)
    clf1.fit(X_train, y_train)
    print('KNeighborsClassifier score 50d', clf1.score(X_test, y_test))
    clf2.fit(X_train, y_train)
    print('DecisionTreeClassifier score 50d',
          clf2.score(X_test, y_test))
    clf3.fit(X_train, y_train)
    print('ExtraTreeClassifier score 50d',
          clf3.score(X_test, y_test))
    clf4.fit(X_train, y_train)
    print('MLPClassifier score 50d',
          clf4.score(X_test, y_test))




    clf1 = OneVsRestClassifier(KNeighborsClassifier())
    clf2 = OneVsRestClassifier(DecisionTreeClassifier())
    clf3 = OneVsRestClassifier(ExtraTreeClassifier())
    clf4 = OneVsRestClassifier(MLPClassifier())
    clf5 = OneVsOneClassifier(NuSVC())
    clf6 = OneVsRestClassifier(LinearSVC())

    from sklearn.linear_model import SGDClassifier
    from sklearn.linear_model import Perceptron
    from sklearn.linear_model import PassiveAggressiveClassifier
    clf7 = OneVsRestClassifier(SGDClassifier())
    clf8 = OneVsRestClassifier(Perceptron())
    clf9 = OneVsRestClassifier(PassiveAggressiveClassifier())

    print('One vs Rest methods case::')
    print('KNeighborsClassifier score 50d',
          clf1.fit(X_train, y_train).score(X_test, y_test))
    print('DecisionTreeClassifier score 50d',
          clf2.fit(X_train, y_train).score(X_test, y_test))
    print('ExtraTreeClassifier score 50d',
          clf3.fit(X_train, y_train).score(X_test, y_test))
    print('MLPClassifier score 50d',
          clf4.fit(X_train, y_train).score(X_test, y_test))

    print('SGDClassifier score 50d',
          clf7.fit(X_train, y_train).score(X_test, y_test))
    print('Perceptron score 50d',
          clf8.fit(X_train, y_train).score(X_test, y_test))
    print('PassiveAggressiveClassifier score 50d',
          clf9.fit(X_train, y_train).score(X_test, y_test))

    print('NuSVC score 50d',
          clf5.fit(X_train, y_train).score(X_test, y_test))
    print('LinearSVC score 50d',
          clf6.fit(X_train, y_train).score(X_test, y_test))

    clf5nu.fit(X_train, y_train)
    print('NuSVC score 50d',
          clf5nu.score(X_test, y_test))
    clf6lin.fit(X_train, y_train)
    print('LinearSVC score 50d',
          clf6lin.score(X_test, y_test))

    from sklearn.datasets import make_friedman1
    from sklearn.feature_selection import RFECV
    from sklearn.neighbors import KNeighborsClassifier
    estimator = DecisionTreeClassifier()
    #selector = RFECV(estimator, step=1, cv=5)
    #selector = selector.fit(wvec, answer)
    #print('feature selection')
    #print(selector.support_)
    #print(selector.ranking_)






if __name__ == '__main__':
    import time

    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))