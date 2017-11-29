from nltk.tokenize import word_tokenize
import string
import gensim
import numpy as np
from collections import Counter
import nltk
import time
import cPickle as pickle
from nltk.corpus import wordnet as wn


path = 'data/data_test'  # whole sentence data
test_30_sentence = 'data/stem_testdata'  # only 30 sentence for function writing


cword_tag = ['NN', 'NNS', 'NNP', 'NNPS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WRB',
'JJ', 'JJR', 'JJS']
fnword_tag = ['PRP','PRP', 'WP', 'WP$', 'IN', 'CC', 'DT', 'PDT', 'WDT', 'CD', 'MD', 'TO']
punct_tag = ['.', ',', ':', "''"]


# return utf-8 h1, h2, ref
def sentences(filepath):
    with open(filepath) as f:
        # with codecs.open(opts.input, 'r', encoding='utf-8') as f:
        for pair in f:
            yield [sentence.strip().decode('utf-8') for sentence in pair.split(' ||| ')]


def string_feature(tokens, ref_tok):
    # tokens = word_tokenize(sentence)
    # ref_tok = word_tokenize(ref)

    # getting 1-4 grams precision, recall, f-measure, and average precision
    ngram_features = [precision(tokens, ref_tok, i) for i in range(1,5)]

    re_arrange = []
    for a, b, c, d in zip(ngram_features[0], ngram_features[1]
            , ngram_features[2], ngram_features[3]):
        re_arrange.extend([a,b,c,d])

    # return with averaged 1-4 gram precisions
    return re_arrange+[np.mean(re_arrange[:4])]


def word_count_feature(pos_h, pos_ref):
    h_count = Counter([x[1] for x in pos_h])
    ref_count = Counter([x[1] for x in pos_ref])

    # count numbers of function, content, puctuation words
    h_fn = sum([cnt for tag, cnt in h_count.items() if tag in fnword_tag])
    h_cont = sum([cnt for tag, cnt in h_count.items() if tag in cword_tag])
    h_punc = sum([cnt for tag, cnt in h_count.items() if tag in punct_tag])

    # count numbers of function, content, puctuation words
    ref_fn = sum([cnt for tag, cnt in ref_count.items() if tag in fnword_tag])
    ref_cont = sum([cnt for tag, cnt in ref_count.items() if tag in cword_tag])
    ref_punc = sum([cnt for tag, cnt in ref_count.items() if tag in punct_tag])

    # length difference of hypothesis and reference is normalized by reference
    #  length
    # assume reference is always not empty
    ref_length = float(len(pos_ref))
    word_count = float(abs(len(pos_ref)-len(pos_h))) / ref_length
    fnd_cnt = float(abs(ref_fn - h_fn)) / ref_length
    cont_cnt = float(abs(ref_cont - h_cont)) / ref_length
    punc_cnt = float(abs(ref_punc - h_punc)) / ref_length

    return [word_count, fnd_cnt, punc_cnt, cont_cnt]


def precision(hypo, ref, n):
    # only 1 reference is used in our case
    # ref: reference sentence. list
    # hypo: hypothesis. list
    # n: degree of ngram
    # return numpy array of precision, recall, and f_measure

    # get and count ngrams in hypothesis
    counts = Counter(ngrams(hypo, n))
    # if there is none, then precision will be zero
    if not counts:
        # return Fraction(0)
        # Since there is no ngram for hypthesis, 0 prediction and 0 correct
        # Thus, 0 accuracy
        return np.array([0, 0, 0])
    # get ngrams of reference and count their appearance.
    # since we have only 1 reference, don't need extra work than this.
    max_counts = Counter(ngrams(ref, n))
    # for each ngrams of hypothesis, find their number of appearances
    # from max_counts, which is from reference.
    clipped_counts = {ngram: min(count, max_counts[ngram]) for ngram, count in counts.items()}
    # return modified precision = sum of intersection appearances divided by
    # sum of candidate appearances

    intersect = sum(clipped_counts.values()) + 1
    predicted = sum(counts.values()) + 1
    wait_tobe_found = sum(max_counts.values()) + 1
    # return numpy array as precision, recall, f_measure
    # we can do add 1 smoothing when calculate above three
    if intersect == 0 or predicted == 0:
        prec = 0.0
    else:
        prec = float(intersect) / float(predicted)

    if intersect == 0 or wait_tobe_found == 0:
        recall = 0.0
    else:
        recall = float(intersect)/float(wait_tobe_found)

    if prec == 0 and recall == 0:
        f_measure = 0.0
    else:
        f_measure = 2*((prec*recall) / (prec+recall))

    return np.array([prec, recall, f_measure])


def word_length(sen):
    # get tokenized sentence, and return followings
    # average word length, type/token ratio
    # number of short word <=3
    # number of long word >=4

    # get number of unique pos tags and divide by number of tokens
    short_word = 0
    long_word = 0
    all_word_len = 0
    all_type = []
    sen_len = float(len(sen))

    # type_token_ratio = len(set([x[1] for x in sen])) / float(len(sen))
    for x in sen:
        all_word_len += len(x[0])
        all_type.append(x[1])
        # count short and long word
        if len(x[0]) <= 3:
            short_word += 1
        else:
            long_word += 1

    # calculate average word length and type/token ratio
    avg_word_len = all_word_len / sen_len
    type_token_ratio = len(set(all_type)) / sen_len

    return [avg_word_len, type_token_ratio, short_word, long_word]


class frequency_check():

    def __init__(self, path):
        # occurance counter for each grams
        self.g1, self.g2, self.g3, self.g4 = occurance_count(path)
        self.g1_len = float(len(self.g1))
        self.g2_len = float(len(self.g2))
        self.g3_len = float(len(self.g3))
        self.g4_len = float(len(self.g4))
        # dictionary of key:gram, value: number of occurance pair
        # divided into 1-4 quartile, for each gram
        # f1 = 1 gram. f1[1] = dictionary for 1 gram 1 quartile
        # f1[1][some 1gram] = number of occurance
        self.f1, self.f2, self.f3, self.f4 = quart_gen(self.g1, self.g2, self.g3, self.g4)
        self.f1_lens = {k: float(sum([vi for vi in v.values()])) for k, v in self.f1.items()}
        self.f2_lens = {k: float(sum([vi for vi in v.values()])) for k, v in self.f2.items()}
        self.f3_lens = {k: float(sum([vi for vi in v.values()])) for k, v in self.f3.items()}
        self.f4_lens = {k: float(sum([vi for vi in v.values()])) for k, v in self.f4.items()}

    def percen_from_all(self, sen, n):
        # sen: list of ngrams
        # n: degree of ngram

        # sum of all words appearance in sentence / all appearance
        if n == 1:
            return sum([self.g1[x] for x in sen]) / self.g1_len
        elif n == 2:
            return sum([self.g2[x] for x in sen]) / self.g2_len
        elif n == 3:
            return sum([self.g3[x] for x in sen]) / self.g3_len
        elif n == 4:
            return sum([self.g4[x] for x in sen]) / self.g4_len
        else:
            # wrong input for ngram degree
            return 0

    def percen_from_freq(self, sen, n, fdegre):
        # sen: list of ngrams
        # n: degree of ngram
        # fdegree: degree of frequency quartile

        # if word or ngram is in quartile dictionary, sum their values
        # and divide by pre-calculated entire-number for each quartile
        if n == 1:
            return sum([self.f1[fdegre][x] for x in sen if x in self.f1[fdegre]]) / self.f1_lens[fdegre]
        elif n == 2:
            return sum([self.f2[fdegre][x] for x in sen if x in self.f2[fdegre]]) / self.f1_lens[fdegre]
        elif n == 3:
            return sum([self.f3[fdegre][x] for x in sen if x in self.f3[fdegre]]) / self.f1_lens[fdegre]
        elif n == 4:
            return sum([self.f4[fdegre][x] for x in sen if x in self.f4[fdegre]]) / self.f1_lens[fdegre]
        else:
            # wrong input for ngram degree
            return 0


def quart_gen(g1, g2, g3, g4):

    # count appearance of each ngrams and sort them as list
    g1_count = sorted([x for x in g1.items()], key=lambda x : x[1])
    g2_count = sorted([x for x in g2.items()], key=lambda x : x[1])
    g3_count = sorted([x for x in g3.items()], key=lambda x : x[1])
    g4_count = sorted([x for x in g4.items()], key=lambda x : x[1])

    # get index to divide list into 4 regions
    g1_scale = [int(0.25 * x * len(g1_count)) for x in range(0, 5)]
    g2_scale = [int(0.25 * x * len(g2_count)) for x in range(0, 5)]
    g3_scale = [int(0.25 * x * len(g3_count)) for x in range(0, 5)]
    g4_scale = [int(0.25 * x * len(g4_count)) for x in range(0, 5)]

    # make each quartile as dictionary according to index above
    g1_freq = {}
    for i in range(4):
        g1_freq[i + 1] = {x[0]: x[1] for x in
                          g1_count[g1_scale[i]:g1_scale[i + 1]]}
    g2_freq = {}
    for i in range(4):
        g2_freq[i + 1] = {x[0]: x[1] for x in
                          g2_count[g2_scale[i]:g2_scale[i + 1]]}
    g3_freq = {}
    for i in range(4):
        g3_freq[i + 1] = {x[0]: x[1] for x in
                          g3_count[g3_scale[i]:g3_scale[i + 1]]}
    g4_freq = {}
    for i in range(4):
        g4_freq[i + 1] = {x[0]: x[1] for x in
                          g4_count[g4_scale[i]:g4_scale[i + 1]]}

    return g1_freq, g2_freq, g3_freq, g4_freq


def occurance_count(data_path):
    # simply read data, parse sentence, and return counter object
    # for checking their appearance in training data
    path1 = '/home/junlinux/Desktop/CSCI544_Last/hw7/data/data_test'
    gram1 = []
    gram2 = []
    gram3 = []
    gram4 = []
    for h1, h2, ref in sentences(data_path):
        h1_tok = word_tokenize(h1)
        h2_tok = word_tokenize(h2)

        gram1.extend([x for x in ngrams(h1_tok, 1)])
        gram1.extend([x for x in ngrams(h2_tok, 1)])
        gram2.extend([x for x in ngrams(h1_tok, 2)])
        gram2.extend([x for x in ngrams(h2_tok, 2)])
        gram3.extend([x for x in ngrams(h1_tok, 3)])
        gram3.extend([x for x in ngrams(h2_tok, 3)])
        gram4.extend([x for x in ngrams(h1_tok, 4)])
        gram4.extend([x for x in ngrams(h2_tok, 4)])

    return Counter(gram1), Counter(gram2), Counter(gram3), Counter(gram4)


def ngrams(sequence, n):
    # sequence: sequence of words, list
    # n: degree of ngrams

    iter_seq = iter(sequence)
    history = []
    # make first n-1 elements ready
    while n > 1:
        n -= 1
        history.append(next(iter_seq))

    # add Nth element, yield, and get rid of first element
    # repeat the process
    for item in iter_seq:
        history.append(item)
        yield tuple(history)
        del history[0]


# save frequency data in disk
def make_frequency_dic(data_path, file_path):
    f_check = frequency_check(data_path)
    with open(file_path+'freq_dic', 'wb') as output:
        pickle.dump(f_check, output, pickle.HIGHEST_PROTOCOL)


# load frequency data from disk
def load_frequency_dic(file_path):
    with open(file_path, 'rb') as input:
        return pickle.load(input)


# loading word vector file
def load_wordvec(vec_path):
    with open(vec_path, 'rb') as infile:
        result = pickle.load(infile)
    return result


# find matching and non-matching word info
def synonym_divide(split):

    h1_in = []
    h1_out = []
    for hw in split[0]:
        if hw not in split[2]:
            syns = set([j.name() for i in wn.synsets(hw) for j in i.lemmas()])
            if len(syns) != 0:
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
            if len(syns) != 0:
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

    return [h1_in, h1_out, h2_in, h2_out, ref_in, ref_out]
    # return [h1_out, h2_out, ref_in, ref_out]


def main():
    start_time = time.time()

    # generate frequency data and store it to the disk
    # f_check = frequency_check('/home/junlinux/Desktop/CSCI544_Last/hw7/data/data_test')
    # make_frequency_dic('/home/junlinux/Desktop/CSCI544_Last/hw7/data/data_test',
    # '/home/junlinux/Desktop/CSCI544_Last/hw7/data/')

    # load pre-made frequency data
    f_check = load_frequency_dic( '/home/junlinux/Desktop/CSCI544_Last/hw7/data/freq_dic')
    # print("--- %s seconds ---" % (time.time() - start_time))
    # print(type(f_check))
    # print(f_check.g1_len, f_check.g2_len, f_check.f3_lens, f_check.f4_lens)

    # This part reads first half of data and generates features
    # I'm using pre-made vector, so skipping this part now.
    # if you need to use it, uncomment

    all_vector = []
    # load hypothesis sentences from file
    for h1, h2, ref in sentences('data/data_test'):
        # tokenize
        tok_h1 = word_tokenize(h1)
        tok_h2 = word_tokenize(h2)
        tok_ref = word_tokenize(ref)

        # pos-tag parse by nltk
        pos_h1 = nltk.pos_tag(tok_h1)
        pos_h2 = nltk.pos_tag(tok_h2)
        pos_ref = nltk.pos_tag(tok_ref)
        #print(pos_h1)

        h1_feature = []
        h2_feature = []

        # ngram precision, recall, f-measure, avg precision
        h1_feature.extend(string_feature(h1, ref))
        h2_feature.extend(string_feature(h2, ref))
        # words, fuction words, puctuation words, contents words count
        h1_feature.extend(word_count_feature(pos_h1, pos_ref))
        h2_feature.extend(word_count_feature(pos_h2, pos_ref))

        # POS tag feature
        tag_h1 = [x[1] for x in pos_h1]
        tag_h2 = [x[1] for x in pos_h2]
        tag_ref = [x[1] for x in pos_ref]
        # POS feature doesn't include avg precision
        h1_feature.extend(string_feature(tag_h1, tag_ref)[:12])
        h2_feature.extend(string_feature(tag_h2, tag_ref)[:12])

        # print('mixture test')
        # print(string_feature(pos_h1, pos_ref))
        # print(string_feature(pos_h1, pos_ref)[:4])

        # mixture ngram precision
        # cut first four to get only precisions
        h1_feature.extend(string_feature(pos_h1, pos_ref)[:4])
        h2_feature.extend(string_feature(pos_h2, pos_ref)[:4])


        #print('str-tag', np.equal(str_m, tag_m))
        #print('pos-str', np.equal(pos_str, str_m))
        #print('pos-tag', np.equal(pos_str, tag_m))

        # word length feature
        # 1. average word length, 2. type / token ratio
        # 3. number of short word <=3
        # 4. number of long word >=4
        h1_feature.extend(word_length(pos_h1))
        h2_feature.extend(word_length(pos_h2))

        # ng_fre_h1 = []
        # ng_fre_h2 = []

        # getting percentage of appearance as feature
        for i in range(1,5):
            h1_ng = ngrams(tok_h1, i)
            h1_feature.append(f_check.percen_from_all(h1_ng, i))

            h2_ng = ngrams(tok_h2, i)
            h2_feature.append(f_check.percen_from_all(h2_ng, i))

        # getting percentage of ngrams in each quartiles
        for i in range(1,5):
            h1_ng = [x for x in ngrams(tok_h1, i)]
            # ngram list, ngram, quartile
            h1_feature.append(f_check.percen_from_freq(h1_ng, i, 1))
            h1_feature.append(f_check.percen_from_freq(h1_ng, i, 2))
            h1_feature.append(f_check.percen_from_freq(h1_ng, i, 3))
            h1_feature.append(f_check.percen_from_freq(h1_ng, i, 4))

            h2_ng = [x for x in ngrams(tok_h2, i)]
            h2_feature.append(f_check.percen_from_freq(h2_ng, i, 1))
            h2_feature.append(f_check.percen_from_freq(h2_ng, i, 2))
            h2_feature.append(f_check.percen_from_freq(h2_ng, i, 3))
            h2_feature.append(f_check.percen_from_freq(h2_ng, i, 4))

        match_info = [len(x)/float(len(tok_ref)) for x in
                      inter_out([tok_h1, tok_h2, tok_ref])]
        #print(match_info)

        # merge vector for h1 and h2
        all_vector.append(np.array([j for x in zip(h1_feature, h2_feature) for j in x]+match_info))
        #all_vector.extend(match_info)
        #print(all_vector)
        #print('ROSE vectors')
        #print(ngrams(tok_h1, 1))
        #print(len(ng_fre_h1), ng_fre_h1)
        #print(len(ng_fre_h2), ng_fre_h2)
        #print(len(all_vector[0]), all_vector)
        #break






    # vec_path is where feature vector is saved
    vec_path = '/home/junlinux/Desktop/CSCI544_Last/hw7/data/word_vector/' \
               'ROSE_wordlen_appear_freq_matchW-smoothed'
    with open(vec_path, 'wb') as outfile:
        pickle.dump(all_vector, outfile, pickle.HIGHEST_PROTOCOL)
        print('ROSE vector:', len(all_vector), 'vectors SUCCEED!')
    print("--- %s seconds ---" % (time.time() - start_time))

    # this vector is pre-made
    # ROSE + word counts + word appearance % +
    #  1-4 gram frequency % in each quartile
    v1 = '/home/junlinux/Desktop/CSCI544_Last/hw7/data/word_vector/' \
         'ROSE_wordlen_appear_freq_matchW-smoothed'
    # this vector is pre-made by taking means of word vectors of h1 and h2
    v2 = '/home/junlinux/Desktop/CSCI544_Last/hw7/data/word_vector/' \
         'glove.6B.50d_vec'
    # loading vector file
    wv1 = load_wordvec(v1)
    wv2 = load_wordvec(v2)
    # print('wv1', len(wv1[0]), wv1[0])
    # print('wv2', len(wv2[0]), wv2[0])
    # mg = np.concatenate((wv1[0], wv2[0]), axis=0)
    # print('wv1+wv2', len(mg), mg )

    # merge these two vectors into 1 file
    all_v = []
    for w1, w2 in zip(wv1, wv2):
        all_v.append(np.concatenate((w1, w2), axis=0))

    # vec_path is where merged vector file will be saved
    vec_path = '/home/junlinux/Desktop/CSCI544_Last/hw7/data/word_vector/' \
               'glove6B_ROSE_wordlen_appear_freq_matchW-smoothed'
    # save vector file by pickle library
    with open(vec_path, 'wb') as outfile:
        pickle.dump(all_v, outfile, pickle.HIGHEST_PROTOCOL)
        print('ROSE vector:', len(all_v), 'vectors SUCCEED!')
    # print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()
