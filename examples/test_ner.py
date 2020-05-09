from __future__ import unicode_literals, division, print_function, absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import object, range, next
import logging
import pickle as pkl
import re
import unicodedata
from copy import deepcopy
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.linear_model import LogisticRegression as logreg

from conec import word2vec
from conec import context2vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def clean_conll2003(text, to_lower=False):
    # clean the text: no fucked up characters
    nfkd_form = unicodedata.normalize("NFKD", text)
    text = nfkd_form.encode("ASCII", "ignore").decode("ASCII")
    # normalize numbers
    text = re.sub(r"[0-9]", "1", text)
    if to_lower:
        text = text.lower()
    return text


class CoNLL2003(object):
    # collected 20102 unique words from a corpus of 218609 words and 946 sentences
    # generator for the conll2003 training data

    def __init__(self, to_lower=False, sources=["data/conll2003/ner/eng.train"]):
        self.sources = sources
        self.to_lower = to_lower

    def __iter__(self):
        """Iterate through all news articles."""
        for fname in self.sources:
            tokens = []
            for line in open(fname):
                if line.startswith("-DOCSTART- -X- -X-"):
                    if tokens:
                        yield tokens
                    tokens = []
                elif line.strip():
                    tokens.append(clean_conll2003(line.split()[0], self.to_lower))
                else:
                    tokens.append('')
            yield tokens


def train_word2vec(train_all=False, it=20, seed=1):
    # train all models for 20 iterations
    # train the word2vec model on a) the training data
    sentences = CoNLL2003(to_lower=True)

    def save_model(model, saven):
        # delete the huge stupid table again
        table = deepcopy(model.table)
        model.table = None
        # pickle the entire model to disk, so we can load&resume training later
        pkl.dump(model, open("data/%s" % saven, 'wb'), -1)
        # reinstate the table to continue training
        model.table = table

    # train the cbow model; default window=5
    alpha = 0.02
    model = word2vec.Word2Vec(sentences, min_count=1, mtype='cbow', hs=0, neg=13, vector_size=200, alpha=alpha, min_alpha=alpha, seed=seed)
    for i in range(1, it):
        print("####### ITERATION %i ########" % (i + 1))
        if not i % 5:
            save_model(model, "conll2003_train_cbow_200_hs0_neg13_seed%i_it%i.model" % (seed, i))
            alpha /= 2.
            alpha = max(alpha, 0.0001)
        model.train(sentences, alpha=alpha, min_alpha=alpha)
    save_model(model, "conll2003_train_cbow_200_hs0_neg13_seed%i_it%i.model" % (seed, it))
    if train_all:
        # and b) the training + test data
        sentences = CoNLL2003(to_lower=True, sources=[
                              "data/conll2003/ner/eng.train", "data/conll2003/ner/eng.testa", "data/conll2003/ner/eng.testb"])
        model = word2vec.Word2Vec(sentences, min_count=1, mtype='cbow', hs=0, neg=13, vector_size=200, seed=seed)
        for i in range(19):
            model.train(sentences)
        # delete the huge stupid table again
        model.table = None
        # pickle the entire model to disk, so we can load&resume training later
        saven = "conll2003_test_20it_cbow_200_hs0_neg13_seed%i.model" % seed
        print("saving model")
        pkl.dump(model, open("data/%s" % saven, 'wb'), -1)


def make_wordfeat(w):
    return [int(w.isalnum()), int(w.isalpha()), int(w.isdigit()),
            int(w.islower()), int(w.istitle()), int(w.isupper()),
            len(w)]


def make_featmat_wordfeat(tokens):
    # tokens: list of words
    return np.array([make_wordfeat(t) for t in tokens])


class ContextEnc_NER(object):

    def __init__(self, w2v_model, contextm=False, sentences=[], w_local=0.4, context_global_only=False, include_wf=False, to_lower=True, normed=True, renorm=True):
        self.clf = None
        self.w2v_model = w2v_model
        self.rep_idx = {word: i for i, word in enumerate(w2v_model.wv.index2word)}
        self.include_wf = include_wf
        self.to_lower = to_lower
        self.w_local = w_local  # how much the local context compared to the global should count
        self.context_global_only = context_global_only  # if only global context should count (0 if global not available -- not same as w_local=0)
        self.normed = normed
        self.renorm = renorm
        # should we include the context?
        if contextm:
            # sentences: depending on what the word2vec model was trained
            self.context_model = context2vec.ContextModel(
                sentences, min_count=1, window=w2v_model.window, wordlist=w2v_model.wv.index2word)
            # --> create a global context matrix
            self.context_model.featmat = self.context_model.get_context_matrix(False, 'max')
        else:
            self.context_model = None

    def make_featmat_rep(self, tokens, local_context_mat=None, tok_idx={}):
        """
        Inputs:
            - tokens: list of words
        Returns:
            - featmat: dense feature matrix for every token
        """
        # possibly preprocess tokens
        if self.to_lower:
            pp_tokens = [t.lower() for t in tokens]
        else:
            pp_tokens = tokens
        dim = self.w2v_model.wv.vector_size
        if self.include_wf:
            dim += 7
        featmat = np.zeros((len(tokens), dim), dtype=float)
        # index in featmat for all known tokens
        idx_featmat = [i for i, t in enumerate(pp_tokens) if t in self.rep_idx]
        if self.normed:
            rep_mat = deepcopy(self.w2v_model.wv.vectors_norm)
        else:
            rep_mat = deepcopy(self.w2v_model.vectors)
        if self.context_model:
            if self.context_global_only:
                # make context matrix out of global context vectors only
                context_mat = lil_matrix((len(tokens), len(self.rep_idx)))
                global_tok_idx = [self.rep_idx[t] for t in pp_tokens if t in self.rep_idx]
                context_mat[idx_featmat, :] = self.context_model.featmat[global_tok_idx, :]
            else:
                # compute the local context matrix
                if not tok_idx:
                    local_context_mat, tok_idx = self.context_model.get_local_context_matrix(pp_tokens)
                local_tok_idx = [tok_idx[t] for t in pp_tokens]
                context_mat = lil_matrix(local_context_mat[local_tok_idx, :])
                assert context_mat.shape == (len(tokens), len(self.rep_idx)), "context matrix has wrong shape"
                # average it with the global context vectors if available
                local_global_tok_idx = [tok_idx[t] for t in pp_tokens if t in self.rep_idx]
                global_tok_idx = [self.rep_idx[t] for t in pp_tokens if t in self.rep_idx]
                context_mat[idx_featmat, :] = self.w_local * lil_matrix(local_context_mat[local_global_tok_idx, :]) + (
                    1. - self.w_local) * self.context_model.featmat[global_tok_idx, :]
            # multiply context_mat with rep_mat to get featmat (+ normalize)
            featmat[:, 0:rep_mat.shape[1]] = csr_matrix(context_mat) * rep_mat
            # length normalize the feature vectors
            if self.renorm:
                fnorm = np.linalg.norm(featmat, axis=1)
                featmat[fnorm > 0, :] = featmat[fnorm > 0, :] / np.array([fnorm[fnorm > 0]]).T
        else:
            # we set the feature matrix with the word2vec embeddings directly;
            # tokens not in the original vocab will have a zero representation
            idx_repmat = [self.rep_idx[t] for t in pp_tokens if t in self.rep_idx]
            featmat[idx_featmat, 0:rep_mat.shape[1]] = rep_mat[idx_repmat, :]
        if self.include_wf:
            featmat[:, dim - 7:] = make_featmat_wordfeat(tokens)
        return featmat

    def train_clf(self, trainfiles):
        # tokens: list of words, labels: list of corresponding labels
        # go document by document because of local context
        final_labels = []
        featmat = []
        for trainfile in trainfiles:
            for tokens, labels in yield_tokens_labels(trainfile):
                final_labels.extend(labels)
                featmat.append(self.make_featmat_rep(tokens))
        featmat = np.vstack(featmat)
        print("training classifier")
        clf = logreg(class_weight='balanced', random_state=1)
        clf.fit(featmat, final_labels)
        self.clf = clf

    def find_ne_in_text(self, text, local_context_mat=None, tok_idx={}):
        featmat = self.make_featmat_rep(text.strip().split(), local_context_mat, tok_idx)
        labels = self.clf.predict(featmat)
        # stitch text back together
        results = []
        for i, t in enumerate(text.strip().split()):
            if results and labels[i] == results[-1][1]:
                results[-1] = (results[-1][0] + " " + t, results[-1][1])
            else:
                if results:
                    results.append((' ', 'O'))
                results.append((t, labels[i]))
        return results


def process_wordlabels(word_labels):
    # process labels
    tokens = []
    labels = []
    for word, l in word_labels:
        if word:
            if l.startswith("I-") or l.startswith("B-"):
                l = l[2:]
            tokens.append(word)
            labels.append(l)
    assert len(tokens) == len(labels), "must have same number of tokens as labels"
    return tokens, labels


def get_tokens_labels(trainfile):
    # read in trainfile to generate training labels
    with open(trainfile) as f:
        word_labels = [(clean_conll2003(line.split()[0]), line.strip().split()[-1]) if line.strip()
                       else ('', 'O') for line in f if not line.startswith("-DOCSTART- -X- -X-")]
    return process_wordlabels(word_labels)


def yield_tokens_labels(trainfile):
    # generate tokens and labels for every document
    word_labels = []
    for line in open(trainfile):
        if line.startswith("-DOCSTART- -X- -X-"):
            if word_labels:
                yield process_wordlabels(word_labels)
            word_labels = []
        elif line.strip():
            word_labels.append((clean_conll2003(line.split()[0]), line.strip().split()[-1]))
        else:
            word_labels.append(('', 'O'))
    yield process_wordlabels(word_labels)


def ne_results_2_labels(ne_results):
    """
    helper function to transform a list of substrings and labels
    into a list of labels for every (white space separated) token
    """
    l_list = []
    last_l = ''
    for i, (substr, l) in enumerate(ne_results):
        if substr == ' ':
            continue
        if not l or l == 'O':
            l_out = 'O'
        elif l == last_l:
            l_out = "B-" + l
        else:
            l_out = "I-" + l
        last_l = l
        if (not i) or (substr.startswith(' ') or ne_results[i - 1][0].endswith(' ')):
            l_list.append(l_out)
        # if there is no space between the previous and last substring, first token gets label
        # of longer subsubstr (i.e. either previous or current)
        elif i and len(ne_results[i - 1][0].split()[-1]) < len(substr.split()[0]):
            l_list.pop()
            l_list.append(l_out)
        l_list.extend([l_out for n in range(len(substr.split()) - 1)])
    return l_list


def apply_conll2003_ner(ner, testfile, outfile):
    """
    Inputs:
        - ner: named entity classifier with find_ne_in_text method
        - testfile: path to the testfile
        - outfile: where the output should be saved
    """
    documents = CoNLL2003(sources=[testfile], to_lower=True)
    documents_it = documents.__iter__()
    local_context_mat, tok_idx = None, {}
    # read in test file + generate outfile
    with open(outfile, 'w') as f_out:
        # collect all the words in a sentence and save other rest of the lines
        to_write, tokens = [], []
        doc_tokens = []
        for line in open(testfile):
            if line.startswith("-DOCSTART- -X- -X-"):
                f_out.write("-DOCSTART- -X- -X- O O\n")
                # we're at a new document, time for a new local context matrix
                if ner.context_model:
                    doc_tokens = next(documents_it)
                    local_context_mat, tok_idx = ner.context_model.get_local_context_matrix(doc_tokens)
            # outfile: testfile + additional column with predicted label
            elif line.strip():
                to_write.append(line.strip())
                tokens.append(clean_conll2003(line.split()[0]))
            else:
                # end of sentence: find all named entities!
                if to_write:
                    ne_results = ner.find_ne_in_text(" ".join(tokens), local_context_mat, tok_idx)
                    assert " ".join(tokens) == "".join(r[0]
                                                       for r in ne_results), "returned text doesn't match"  # sanity check
                    l_list = ne_results_2_labels(ne_results)
                    assert len(l_list) == len(tokens), "Error: %i labels but %i tokens" % (len(l_list), len(tokens))
                    for i, line in enumerate(to_write):
                        f_out.write(to_write[i] + " " + l_list[i] + "\n")
                to_write, tokens = [], []
                f_out.write("\n")


def log_results(clf_ner, description, filen='', subf=''):
    import os
    if not os.path.exists('data/conll2003_results'):
        os.mkdir('data/conll2003_results')
    if not os.path.exists('data/conll2003_results%s' % subf):
        os.mkdir('data/conll2003_results%s' % subf)
    import subprocess
    print("applying to training set")
    apply_conll2003_ner(clf_ner, 'data/conll2003/ner/eng.train', 'data/conll2003_results%s/eng.out_train.txt' % subf)
    print("applying to test set")
    apply_conll2003_ner(clf_ner, 'data/conll2003/ner/eng.testa', 'data/conll2003_results%s/eng.out_testa.txt' % subf)
    apply_conll2003_ner(clf_ner, 'data/conll2003/ner/eng.testb', 'data/conll2003_results%s/eng.out_testb.txt' % subf)
    # write out results
    with open('data/conll2003_results/output_all_%s.txt' % filen, 'a') as f:
        f.write('%s\n' % description)
        f.write('results on training data\n')
        out = subprocess.getstatusoutput('data/conll2003/ner/bin/conlleval < data/conll2003_results%s/eng.out_train.txt' % subf)[1]
        f.write(out)
        f.write('\n')
        f.write('results on testa\n')
        out = subprocess.getstatusoutput('data/conll2003/ner/bin/conlleval < data/conll2003_results%s/eng.out_testa.txt' % subf)[1]
        f.write(out)
        f.write('\n')
        f.write('results on testb\n')
        out = subprocess.getstatusoutput('data/conll2003/ner/bin/conlleval < data/conll2003_results%s/eng.out_testb.txt' % subf)[1]
        f.write(out)
        f.write('\n')
        f.write('\n')


if __name__ == '__main__':
    seed = 3
    it = 20
    train_word2vec(train_all=False, it=it, seed=seed)
    # load pretrained word2vec model
    with open("data/conll2003_train_cbow_200_hs0_neg13_seed%i_it%i.model" % (seed, it), 'rb') as f:
        w2v_model = pkl.load(f)
    # train a classifier with these word embeddings on the training part
    clf_ner = ContextEnc_NER(w2v_model, include_wf=False)
    clf_ner.train_clf(['data/conll2003/ner/eng.train'])
    # apply the classifier to all training and test parts of the CoNLL2003 task,
    # run the evaluation script and save the results
    log_results(clf_ner, '####### word2vec model, seed: %i, it: %i' % (seed, it), 'word2vec_%i' % seed, '_word2vec_%i_%i' % (seed, it))
    """
        results on training data
        processed 204567 tokens with 23499 phrases; found: 38310 phrases; correct: 11537.
        accuracy:  84.48%; precision:  30.11%; recall:  49.10%; FB1:  37.33
                      LOC: precision:  51.57%; recall:  75.06%; FB1:  61.14  10391
                     MISC: precision:  21.22%; recall:  39.70%; FB1:  27.66  6432
                      ORG: precision:  18.52%; recall:  29.08%; FB1:  22.63  9924
                      PER: precision:  25.73%; recall:  45.08%; FB1:  32.76  11563
        results on testa
        processed 51578 tokens with 5942 phrases; found: 8422 phrases; correct: 2525.
        accuracy:  84.04%; precision:  29.98%; recall:  42.49%; FB1:  35.16
                      LOC: precision:  52.03%; recall:  66.85%; FB1:  58.52  2360
                     MISC: precision:  25.25%; recall:  41.54%; FB1:  31.41  1517
                      ORG: precision:  19.26%; recall:  30.28%; FB1:  23.54  2108
                      PER: precision:  20.85%; recall:  27.58%; FB1:  23.74  2437
        results on testb
        processed 46666 tokens with 5648 phrases; found: 7338 phrases; correct: 1960.
        accuracy:  82.26%; precision:  26.71%; recall:  34.70%; FB1:  30.19
                      LOC: precision:  52.07%; recall:  66.49%; FB1:  58.40  2130
                     MISC: precision:  19.05%; recall:  38.32%; FB1:  25.45  1412
                      ORG: precision:  19.64%; recall:  22.40%; FB1:  20.93  1894
                      PER: precision:  11.04%; recall:  12.99%; FB1:  11.94  1902
    """

    # load the text again (same as word2vec model was trained on) to generate the context matrix
    sentences = CoNLL2003(to_lower=True)
    # only use global context; no rep for out-of-vocab
    clf_ner = ContextEnc_NER(w2v_model, contextm=True, sentences=sentences, w_local=0., context_global_only=True)
    clf_ner.train_clf(['data/conll2003/ner/eng.train'])
    # evaluate the results again
    log_results(clf_ner, '####### context enc with global context matrix only, seed: %i, it: %i' % (seed, it), 'conec_global_%i' % seed, '_conec_global_%i_%i' % (seed, it))

    # for the out-of-vocabulary words in the dev and test set, only the local context matrix (based on only the current doc)
    # is used to generate the respective word embeddings; where a global context vector is available (for all words in the training set)
    # we use a combination of the local and global context, determined by w_local
    for w_local in [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]:
        print(w_local)
        clf_ner = ContextEnc_NER(w2v_model, contextm=True, sentences=sentences, w_local=w_local)
        clf_ner.train_clf(['data/conll2003/ner/eng.train'])
        # evaluate the results again
        log_results(clf_ner, '####### context enc with a combination of the global and local context matrix (w_local=%.1f), seed: %i, it: %i' % (w_local, seed, it), 'conec_%i_%i' % (round(w_local*10), seed), '_conec_%i_%i_%i' % (round(w_local*10), seed, it))
        """
            results on training data
            processed 204567 tokens with 23499 phrases; found: 33708 phrases; correct: 11675.
            accuracy:  84.34%; precision:  34.64%; recall:  49.68%; FB1:  40.82
                          LOC: precision:  57.46%; recall:  75.34%; FB1:  65.20  9361
                         MISC: precision:  19.56%; recall:  37.14%; FB1:  25.62  6530
                          ORG: precision:  19.16%; recall:  24.62%; FB1:  21.55  8119
                          PER: precision:  35.71%; recall:  52.47%; FB1:  42.50  9698
            results on testa
            processed 51578 tokens with 5942 phrases; found: 8756 phrases; correct: 3244.
            accuracy:  85.01%; precision:  37.05%; recall:  54.59%; FB1:  44.14
                          LOC: precision:  56.96%; recall:  77.74%; FB1:  65.75  2507
                         MISC: precision:  22.97%; recall:  41.76%; FB1:  29.64  1676
                          ORG: precision:  20.96%; recall:  28.64%; FB1:  24.20  1832
                          PER: precision:  38.20%; recall:  56.84%; FB1:  45.69  2741
            results on testb
            processed 46666 tokens with 5648 phrases; found: 8407 phrases; correct: 2830.
            accuracy:  84.17%; precision:  33.66%; recall:  50.11%; FB1:  40.27
                          LOC: precision:  53.21%; recall:  74.58%; FB1:  62.11  2338
                         MISC: precision:  16.29%; recall:  36.32%; FB1:  22.50  1565
                          ORG: precision:  24.44%; recall:  30.04%; FB1:  26.95  2042
                          PER: precision:  33.79%; recall:  51.45%; FB1:  40.79  2462
        """
