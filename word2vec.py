from nltk.tokenize import word_tokenize
import string
import gensim
import numpy as np
import cPickle as pickle

mean_vec = 1
mean_tf = 2
mean_tf_whole = 3

path = 'data/data_test'  # whole sentence data
test_30_sentence = 'data/stem_testdata'  # only 30 sentence for function writing
v_path = "data/glove.6B/glove.6B.50d.txt"


def sentences2():
    with open(path) as f:
        # with codecs.open(opts.input, 'r', encoding='utf-8') as f:
        for pair in f:
            yield [sentence.strip().split() for sentence in pair.split(' ||| ')]

def sentences(filepath):
    with open(filepath) as f:
        # with codecs.open(opts.input, 'r', encoding='utf-8') as f:
        for pair in f:
            yield [sentence.strip(string.punctuation).decode('utf-8') for sentence in pair.split(' ||| ')]

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        return self

    def transform(self, X):
        #print('X', len(X))

        result =  np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

        return result


from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(word2vec.itervalues().next())

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
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])


def word_vec_load(vpath="data/glove.6B/glove.6B.50d.txt"):
    with open(vpath, "rb") as lines:
        w2v = {line.split()[0]: np.array(map(float, line.split()[1:]))
               for line in lines}
        return w2v


def tftrain(filepath, glove_vec):
    tfvec = TfidfEmbeddingVectorizer(glove_vec)
    words = []
    for h1, h2, ref in sentences(filepath):
        words.extend([x.strip(string.punctuation) for x in h1.split()])
        words.extend([x.strip(string.punctuation) for x in h2.split()])
        words.extend([x.strip(string.punctuation) for x in ref.split()])
    tfvec.fit(words,[])
    return tfvec


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

        # vectype 1 - average of all word vectors
        if vectype==1:
            means = vectorizer.transform(split)
            all_mean.append(np.concatenate((means[0], means[1], means[2]), axis=0))

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
    if len(all_mean)>0:
        with open(vec_path, 'wb') as outfile:
            pickle.dump(all_mean, outfile, pickle.HIGHEST_PROTOCOL)
            print(vecfile, vectype, len(all_mean), 'vectors SUCCEED!')
    else:
        print(vecfile, vectype, 'failed to word vec generation.')
        print('vectype 1 - average, 2 - tf-idf by one line, 3 - tf-idf by whole data')


def main():
    # filepath: sentence data file path
    # vecfile: word vector file path pre-generated from other
    # vectype: compression methods. Average, avg+tf-idf one line, agg+tf-idf whole data
    # vec_path: vector file save path

    filepath =  '/home/junpo02/Desktop/CSCI544/hw7_2/data/stem_testdata' # 'data/data_test'
    vecfile = '/home/junpo02/Desktop/CSCI544/hw7_2/data/glove.6B/glove.6B.50d.txt'

    vec_files = ['/home/junpo02/Desktop/CSCI544/hw7_2/data/glove.6B/glove.6B.50d.txt'
     ,'/home/junpo02/Desktop/CSCI544/hw7_2/data/glove.6B/glove.6B.100d.txt'
     ,'/home/junpo02/Desktop/CSCI544/hw7_2/data/glove.6B/glove.6B.200d.txt'
     ,'/home/junpo02/Desktop/CSCI544/hw7_2/data/glove.6B/glove.6B.300d.txt'
     ,'/home/junpo02/Desktop/CSCI544/hw7_2/data/glove.42B.300d.txt'
     ,'/home/junpo02/Desktop/CSCI544/hw7_2/data/glove.840B.300d.txt'
                 ]
    vec_path = '/home/junpo02/Desktop/CSCI544/hw7_2/data/word_vector/'



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




if __name__ == '__main__':
    import time

    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
