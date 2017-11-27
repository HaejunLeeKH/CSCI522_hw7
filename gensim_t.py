from nltk.tokenize import word_tokenize
import string
import gensim
import numpy as np

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

def main():


    glove_vec = word_vec_load(v_path)  # load word vector
    MeanVectorizer = MeanEmbeddingVectorizer(glove_vec)  # create meanvectorizer
    tfvec = TfidfEmbeddingVectorizer(glove_vec)

    te = tftrain(path, glove_vec)
    print(len(te.word2weight))
    print(len(te.word2weight.keys()))
    #exit()


    cnt = 0
    all_mean = []
    all_meantf = []
    for h1, h2, ref in sentences(test_30_sentence):

        cnt += 1
        if cnt > 2:
            break

        split = [[x.strip(string.punctuation) for x in h1.split()],
                 [x.strip(string.punctuation) for x in h2.split()],
                 [x.strip(string.punctuation) for x in ref.split()]
                 ]
        #h1_split = [x.strip(string.punctuation) for x in h1.split()]
        h1_mean = MeanVectorizer.transform(split)
        all_mean.append(np.concatenate((h1_mean[0], h1_mean[1], h1_mean[2]), axis=0))
        #print('h1 split', h1.split())
        #print('h1 split+punc', h1_split)
        print('h1_mean', len(h1_mean))
        print('split', len(split))
        print('h1', len(split[0]), len(h1_mean[0]), split[0])
        print('h2', len(split[1]), len(h1_mean[1]), split[1])
        print('ref', len(split[2]), len(h1_mean[2]), split[2])
        print('len h1_mean', len(h1_mean))
        print(h1_mean)

        tfvec.fit(split, [1])
        h1_tfmean = te.transform(split)
        all_meantf.append(
            np.concatenate((h1_tfmean[0], h1_tfmean[1], h1_tfmean[2]), axis=0))
        # print('h1 split', h1.split())
        # print('h1 split+punc', h1_split)
        print('h1_tfmean', len(h1_tfmean))
        print('split', len(split))
        print('h1', len(split[0]), len(h1_tfmean[0]), split[0])
        print('h2', len(split[1]), len(h1_tfmean[1]), split[1])
        print('ref', len(split[2]), len(h1_tfmean[2]), split[2])
        print('len h1_mean', len(h1_tfmean))
        print(h1_tfmean)


    print len(all_mean)
    print (all_mean[0].shape)
    print(all_mean)
    y = np.random.randint(2, size=(1, 30))
    import random
    y2 = random.sample(range(0, 100), 2)
    from sklearn.naive_bayes import BernoulliNB
    clf = BernoulliNB()


    import cPickle as pickle
    # save and load numpy array to and from file
    # to preserve the order, which is important to match with target
    # using pickle
    with open('/home/junlinux/Desktop/CSCI544_Last/hw7/data/save_test_gen', 'wb') as outfile:
        pickle.dump(all_mean, outfile, pickle.HIGHEST_PROTOCOL)

    # now load from file
    with open('/home/junlinux/Desktop/CSCI544_Last/hw7/data/save_test_gen', 'rb') as infile:
        result = pickle.load(infile)

    print('Does out and load are same?', np.equal(all_mean, result))
    print('all_mean', type(all_mean[0]))
    print('result', type(result[0]))

    clf.fit(result, y2)
    print(y2)
    print(clf.predict(all_mean[:1]))

    # this tf idf encountered divide by zero situation
    # guess it's because calculated tf-idf with each line
    # maybe need to do for whole data set(which is first half)
    print('tf test fit')
    clf.fit(all_meantf, [1,1])
    print([1,1])
    print(clf.predict(all_mean[:1]))

    #print('tfidf test')
    #print(tfvec.word2weight)
    #tfvec.fit(all_mean, y2)
    #print(tfvec.word2weight)



if __name__ == '__main__':
    import time

    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
