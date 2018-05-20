from __future__ import unicode_literals, division, print_function, absolute_import
from builtins import object
from collections import defaultdict
from copy import deepcopy
import numpy as np
from scipy.sparse import csr_matrix, dok_matrix, lil_matrix


class ContextModel(object):

    def __init__(self, sentences, min_count=5, window=5, forward=True, backward=True, wordlist=[], progress=1000):
        """
        sentences: list/generator of lists of words
        in case this is based on a pretrained word2vec model, give the index2word attribute as wordlist

        Attributes:
            - min_count: how often a word has to occur at least
            - window: how many words in a word's context should be considered
            - word2index: {word:idx}
            - index2word: [word1, word2, ...]
            - wcounts: {word: frequency}
            - featmat: n_voc x n_voc sparse array with weighted context word counts for every word
            - progress: after how many sentences a progress printout should occur (default 1000)
        """
        self.progress = progress
        self.min_count = min_count
        self.window = window
        self.build_windex(sentences, wordlist)
        self.forward = forward
        self.backward = backward
        self._get_raw_context_matrix(sentences)

    def build_windex(self, sentences, wordlist=[]):
        """
        go through all the sentences and get an overview of all used words and their frequencies
        """
        # get an overview of the vocabulary
        vocab = defaultdict(int)
        total_words = 0
        for sentence_no, sentence in enumerate(sentences):
            if not sentence_no % self.progress:
                print("PROGRESS: at sentence #%i, processed %i words and %i unique words" % (sentence_no, sum(vocab.values()), len(vocab)))
            for word in sentence:
                vocab[word] += 1
        print("collected %i unique words from a corpus of %i words and %i sentences" % (len(vocab), sum(vocab.values()), sentence_no + 1))
        # assign a unique index to each word and remove all words with freq < min_count
        self.wcounts, self.word2index, self.index2word = {}, {}, []
        if not wordlist:
            wordlist = [word for word, c in vocab.items() if c >= self.min_count]
        for word in wordlist:
            self.word2index[word] = len(self.word2index)
            self.index2word.append(word)
            self.wcounts[word] = vocab[word]

    def _get_raw_context_matrix(self, sentences):
        """
        compute the raw context matrix with weighted counts
        it has an entry for every word in the vocabulary
        """
        # make the feature matrix
        featmat = lil_matrix((len(self.index2word), len(self.index2word)), dtype=float)
        for sentence_no, sentence in enumerate(sentences):
            if not sentence_no % self.progress:
                print("PROGRESS: at sentence #%i" % sentence_no)
            sentence = [word if word in self.word2index else None for word in sentence]
            # forward pass
            if self.forward:
                for i, word in enumerate(sentence[:-1]):
                    if word:
                        # get all words in the forward window
                        wwords = sentence[i + 1:min(i + 1 + self.window, len(sentence))]
                        for j, w in enumerate(wwords, 1):
                            if w:
                                featmat[self.word2index[word], self.word2index[w]] += 1.  # /j
            # backwards pass
            if self.backward:
                sentence_back = sentence[::-1]
                for i, word in enumerate(sentence_back[:-1]):
                    if word:
                        # get all words in the forward window of the backwards sentence
                        wwords = sentence_back[i + 1:min(i + 1 + self.window, len(sentence_back))]
                        for j, w in enumerate(wwords, 1):
                            if w:
                                featmat[self.word2index[word], self.word2index[w]] += 1.  # /j
        print("PROGRESS: through with all the sentences")
        self.featmat = csr_matrix(featmat)

    def get_context_matrix(self, fill_diag=True, norm='count'):
        """
        for every word in the sentences, create a vector that contains the counts of its context words
        (weighted by the distance to it with a max distance of window)
        Inputs:
            - norm: if the feature matrix should be normalized to contain ones on the diagonal
                    (--> average context vectors)
            - fill_diag: if diagonal of featmat should be filled with word counts
        Returns:
            - featmat: n_voc x n_voc sparse array with weighted context word counts for every word
        """
        featmat = deepcopy(self.featmat)
        # fill up the diagonals with the total counts of each word --> similarity matrix
        if fill_diag:
            featmat = lil_matrix(featmat)
            for i, word in enumerate(self.index2word):
                featmat[i, i] = self.wcounts[word]
            featmat = csr_matrix(featmat)
        assert ((featmat - featmat.transpose()).data**2).sum() < 2.220446049250313e-16, "featmat not symmetric"
        # possibly normalize by the max counts
        if norm == 'count':
            print("normalizing feature matrix by word count")
            normmat = lil_matrix(featmat.shape, dtype=float)
            normmat.setdiag([1. / self.wcounts[word] for word in self.index2word])
            featmat = csr_matrix(normmat) * featmat
        elif norm == 'max':
            print("normalizing feature matrix by max counts")
            normmat = lil_matrix(featmat.shape, dtype=float)
            normmat.setdiag([1. / v[0] if v[0] else 1. for v in featmat.max(axis=1).toarray()])
            featmat = csr_matrix(normmat) * featmat
        return featmat

    def get_local_context_matrix(self, tokens, forward=True, backward=True):
        """
        compute a local context matrix. it has an entry for every token, even if it is not present in the vocabulary
        Inputs:
            - tokens: list of words
        Returns:
            - local_featmat: size len(set(tokens)) x n_vocab
            - tok_idx: {word: index} to map the words from the tokens list to an index of the featmat
        """
        # for every token we still only need one representation per document
        tok_idx = {word: i for i, word in enumerate(set(tokens))}
        featmat = lil_matrix((len(tok_idx), len(self.index2word)), dtype=float)
        # clean out context words we don't know
        known_tokens = [word if word in self.word2index else None for word in tokens]
        # forward pass
        if self.forward:
            for i, word in enumerate(tokens[:-1]):
                # get all words in the forward window
                wwords = known_tokens[i + 1:min(i + 1 + self.window, len(known_tokens))]
                for j, w in enumerate(wwords, 1):
                    if w:
                        featmat[tok_idx[word], self.word2index[w]] += 1. / j
        # backwards pass
        if self.backward:
            tokens_back = tokens[::-1]
            known_tokens_back = known_tokens[::-1]
            for i, word in enumerate(tokens_back[:-1]):
                # get all words in the forward window of the backwards sentence, incl. word itself
                wwords = known_tokens_back[i + 1:min(i + 1 + self.window, len(known_tokens_back))]
                for j, w in enumerate(wwords, 1):
                    if w:
                        featmat[tok_idx[word], self.word2index[w]] += 1. / j
        featmat = csr_matrix(featmat)
        # normalize matrix
        normmat = lil_matrix((featmat.shape[0], featmat.shape[0]), dtype=float)
        normmat.setdiag([1. / v[0] if v[0] else 1. for v in featmat.max(axis=1).toarray()])
        featmat = csr_matrix(normmat) * featmat
        return featmat, tok_idx
