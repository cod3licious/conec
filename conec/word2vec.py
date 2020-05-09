# Original Code by Radim Rehurek <me@radimrehurek.com>
# [Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html]
# see: http://radimrehurek.com/gensim/
#
# Rewrite by Franziska Horn <cod3licious@gmail.com>

from __future__ import unicode_literals, division, print_function, absolute_import
from builtins import object, range, str
import time
import logging
import heapq
from copy import deepcopy
from math import sqrt
import numpy as np

logger = logging.getLogger("word2vec")


class Vocab(object):
    """
    A single vocabulary item, used internally e.g. for constructing binary trees
    (incl. both word leaves and inner nodes).

    Possible Fields:
        - count: how often the word occurred in the training sentences
        - index: the word's index in the embedding
    """

    def __init__(self, **kwargs):
        self.count = 0
        self.__dict__.update(kwargs)

    def __lt__(self, other):  # used for sorting in a priority queue
        return self.count < other.count

    def __str__(self):
        vals = ['%s:%r' % (key, self.__dict__[key]) for key in sorted(self.__dict__) if not key.startswith('_')]
        return "%s(%s)" % (self.__class__.__name__, ', '.join(vals))


class Word2VecEmbeddings(object):
    """
    Word2Vec embeddings only - can't be trained further, but enough for all calculations
    """
    def __init__(self, vector_size=100):
        """
        Initialize Word2Vec embeddings

        Inputs:
            - vector_size: (default 100) dimensionality of embedding
        """
        self.vector_size = vector_size
        self.vectors = np.zeros((0, vector_size))
        self.vectors_norm = None
        self.vocab = {}  # mapping from a word (string) to a Vocab object
        self.index2word = []  # map from a word's matrix index (int) to the word (string)

    def __str__(self):
        return "Word2VecEmbeddings(vocab=%s, size=%s)" % (len(self.index2word), self.vector_size)

    def __getitem__(self, word):
        """
        Return a word's representations in vector space, as a 1D numpy array.

        Example:
          >>> trained_model['woman']
          array([ -1.40128313e-02, ...]
        """
        return self.vectors[self.vocab[word].index]

    def __contains__(self, word):
        return word in self.vocab

    def build_vocab(self, sentences, min_count=5, thr=0):
        """
        Build vocabulary from a sequence of sentences (can be a once-only generator stream).
        Each sentence must be a list of strings.

        Inputs:
            - sentences: List or generator object supplying lists of (preprocessed) words
                         used to train the model (otherwise train manually with model.train(sentences))
            - min_count: (default 5) how often a word has to occur at least to be taken into the vocab
            - thr: (default 0) threshold for computing probabilities for sub-sampling words in training
        """
        logger.info("collecting all words and their counts")
        sentence_no, vocab = -1, {}
        total_words = 0
        for sentence_no, sentence in enumerate(sentences):
            if not sentence_no % 10000:
                logger.info("PROGRESS: at sentence #%i, processed %i words and %i unique words" %
                            (sentence_no, total_words, len(vocab)))
            for word in sentence:
                total_words += 1
                try:
                    vocab[word].count += 1
                except KeyError:
                    vocab[word] = Vocab(count=1)
        logger.info("collected %i unique words from a corpus of %i words and %i sentences" %
                    (len(vocab), total_words, sentence_no + 1))
        # assign a unique index to each word
        self.vocab, self.index2word = {}, []
        for word, v in vocab.items():
            if v.count >= min_count:
                v.index = len(self.vocab)
                self.index2word.append(word)
                self.vocab[word] = v
        logger.info("total of %i unique words after removing those with count < %s" % (len(self.vocab), min_count))
        # add probabilities for sub-sampling (if thr > 0)
        if thr > 0:
            total_words = float(sum(v.count for v in self.vocab.values()))
            for word in self.vocab:
                # formula from paper
                # self.vocab[word].prob = max(0.,1.-sqrt(thr*total_words/self.vocab[word].count))
                # formula from code
                self.vocab[word].prob = (sqrt(self.vocab[word].count / (thr * total_words)
                                              ) + 1.) * (thr * total_words) / self.vocab[word].count
        else:
            # if prob is 0, word wont get discarded
            for word in self.vocab:
                self.vocab[word].prob = 0.

    def init_sims(self):
        # for convenience (for later similarity computations, etc.), store all
        # embeddings additionally as unit length vectors
        self.vectors_norm = self.vectors / np.array([np.linalg.norm(self.vectors, axis=1)]).T

    def similarity(self, w1, w2):
        """
        Compute cosine similarity between two words.

        Example::
          >>> trained_model.similarity('woman', 'man')
          0.73723527
        """
        if self.vectors_norm is None:
            self.init_sims()
        return np.inner(self.vectors_norm[self.vocab[w1].index], self.vectors_norm[self.vocab[w2].index])

    def most_similar(self, positive=[], negative=[], topn=10):
        """
        Find the top-N most similar words. Positive words contribute positively towards the
        similarity, negative words negatively.

        This method computes cosine similarity between a simple mean of the projection
        weight vectors of the given words, and corresponds to the `word-analogy` and
        `distance` scripts in the original word2vec implementation.

        Example::
          >>> trained_model.most_similar(positive=['woman', 'king'], negative=['man'])
          [('queen', 0.50882536), ...]
        """
        if self.vectors_norm is None:
            self.init_sims()
        if isinstance(positive, str) and not negative:
            # allow calls like most_similar('dog'), as a shorthand for most_similar(['dog'])
            positive = [positive]

        # add weights for each word, if not already present; default to 1.0 for positive and -1.0 for negative words
        positive = [(word, 1.) if isinstance(word, str) else word for word in positive]
        negative = [(word, -1.) if isinstance(word, str) else word for word in negative]

        # compute the weighted average of all words
        all_words = set()
        mean = np.zeros(self.vector_size)
        for word, weight in positive + negative:
            try:
                mean += weight * self.vectors_norm[self.vocab[word].index]
                all_words.add(self.vocab[word].index)
            except KeyError:
                print("word '%s' not in vocabulary" % word)
        if not all_words:
            raise ValueError("cannot compute similarity with no input")
        dists = np.dot(self.vectors_norm, mean / np.linalg.norm(mean))
        if not topn:
            return dists
        best = np.argsort(dists)[::-1][:topn + len(all_words)]
        # ignore (don't return) words from the input
        result = [(self.index2word[sim], dists[sim]) for sim in best if sim not in all_words]
        return result[:topn]

    def doesnt_match(self, words):
        """
        Which word from the given list doesn't go with the others?

        Example::
          >>> trained_model.doesnt_match("breakfast cereal dinner lunch".split())
          'cereal'
        """
        if self.vectors_norm is None:
            self.init_sims()
        words = [word for word in words if word in self.vocab]  # filter out OOV words
        logger.debug("using words %s" % words)
        if not words:
            raise ValueError("cannot select a word from an empty list")
        # which word vector representation is furthest away from the mean?
        selection = self.vectors_norm[[self.vocab[word].index for word in words]]
        mean = np.mean(selection, axis=0)
        sim = np.dot(selection, mean / np.linalg.norm(mean))
        return words[np.argmin(sim)]


class Word2Vec(object):
    """
    Word2Vec Model, which can be trained and then contains word embedding that can be used for all kinds of cool stuff.
    """

    def __init__(self, sentences=None, vector_size=100, mtype='sg', hs=1, neg=0, window=5,
                 thr=0, min_count=5, alpha=0.025, min_alpha=0.0001, seed=1):
        """
        Initialize Word2Vec model

        Inputs:
            - sentences: (default None) List or generator object supplying lists of (preprocessed) words
                         used to train the model (otherwise train manually with model.train(sentences))
            - vector_size: (default 100) dimensionality of embedding
            - mtype: (default 'sg') type of model: either 'sg' (skipgram) or 'cbow' (bag of words)
            - hs: (default 1) if != 0, hierarchical softmax will be used for training the model
            - neg: (default 0) if > 0, negative sampling will be used for training the model;
                   neg specifies the # of noise words
            - window: (default 5) max distance of context words from target word in training
            - thr: (default 0) threshold for computing probabilities for sub-sampling words in training
            - min_count: (default 5) how often a word has to occur at least to be taken into the vocab
            - alpha: (default 0.025) initial learning rate
            - min_alpha: (default 0.0001) if < alpha, the learning rate will be decreased to min_alpha
            - seed: (default 1) random seed (for initializing the embeddings)
        """
        assert mtype.lower() in ('sg', 'cbow'), "unknown model, use 'sg' or 'cbow'"
        self.wv = Word2VecEmbeddings(vector_size)  # stores the actual word2vec embeddings
        self.mtype = mtype.lower()
        self.hs = hs
        self.neg = neg
        self.window = window
        self.thr = thr
        self.min_count = min_count
        self.alpha = alpha
        self.min_alpha = min_alpha
        self.seed = seed
        # possibly train model
        if sentences:
            self.train_setup(sentences)
            self.train(sentences)

    def __str__(self):
        return "Word2Vec(vocab=%s, size=%s, mtype=%s, hs=%i, neg=%i)" % (len(self.wv.index2word), self.wv.vector_size, self.mtype, self.hs, self.neg)

    def reset_weights(self):
        """
        Reset all projection weights to an initial (untrained) state, but keep the existing vocabulary.
        """
        np.random.seed(self.seed)
        # weights
        self.syn1 = np.asarray(
            np.random.uniform(
                low=-4 * np.sqrt(6. / (len(self.wv.vocab) + self.wv.vector_size)),
                high=4 * np.sqrt(6. / (len(self.wv.vocab) + self.wv.vector_size)),
                size=(len(self.wv.vocab), self.wv.vector_size)
            ),
            dtype=float
        )
        self.syn1neg = np.asarray(
            np.random.uniform(
                low=-4 * np.sqrt(6. / (len(self.wv.vocab) + self.wv.vector_size)),
                high=4 * np.sqrt(6. / (len(self.wv.vocab) + self.wv.vector_size)),
                size=(len(self.wv.vocab), self.wv.vector_size)
            ),
            dtype=float
        )
        # embedding
        self.wv.vectors = np.asarray(
            np.random.uniform(
                low=-4 * np.sqrt(6. / (len(self.wv.vocab) + self.wv.vector_size)),
                high=4 * np.sqrt(6. / (len(self.wv.vocab) + self.wv.vector_size)),
                size=(len(self.wv.vocab), self.wv.vector_size)
            ),
            dtype=float
        )

    def _make_table(self, table_size=100000000., power=0.75):
        """
        Create a table using stored vocabulary word counts for drawing random words in the negative
        sampling training routines.
        """
        vocab_size = len(self.wv.vocab)
        logger.info("constructing a table with noise distribution from %i words" % vocab_size)
        # table (= list of words) of noise distribution for negative sampling
        self.table = np.zeros(int(table_size), dtype=int)
        # compute sum of all power (Z in paper)
        train_words_pow = float(sum([self.wv.vocab[word].count**power for word in self.wv.vocab]))
        # go through the whole table and fill it up with the word indexes proportional to a word's count**power
        widx = 0
        # normalize count^0.75 by Z
        d1 = self.wv.vocab[self.wv.index2word[widx]].count**power / train_words_pow
        for tidx in range(int(table_size)):
            self.table[tidx] = widx
            if tidx / table_size > d1:
                widx += 1
                d1 += self.wv.vocab[self.wv.index2word[widx]].count**power / train_words_pow
            if widx >= vocab_size:
                widx = vocab_size - 1

    def _create_binary_tree(self):
        """
        Create a binary Huffman tree for the hs model using stored vocabulary word counts.
        Frequent words will have shorter binary codes.
        """
        vocab_size = len(self.wv.vocab)
        logger.info("constructing a huffman tree from %i words" % vocab_size)
        # build the huffman tree
        heap = list(self.wv.vocab.values())
        heapq.heapify(heap)
        for i in range(vocab_size - 1):
            min1, min2 = heapq.heappop(heap), heapq.heappop(heap)
            heapq.heappush(heap, Vocab(count=min1.count + min2.count, index=i + vocab_size, left=min1, right=min2))
        # recurse over the tree, assigning a binary code to each vocabulary word
        if heap:
            max_depth, stack = 0, [(heap[0], [], [])]
            while stack:
                node, codes, points = stack.pop()
                if node.index < vocab_size:
                    # leaf node => store its path from the root
                    node.code, node.point = codes, points
                    max_depth = max(len(codes), max_depth)
                else:
                    # inner node => continue recursion
                    points = np.array(list(points) + [node.index - vocab_size], dtype=int)
                    stack.append((node.left, np.array(list(codes) + [0], dtype=int), points))
                    stack.append((node.right, np.array(list(codes) + [1], dtype=int), points))
            logger.info("built huffman tree with maximum node depth %i" % max_depth)

    def train_setup(self, sentences):
        """
        Do a bunch of initializations etc before training can start
        """
        self.wv.build_vocab(sentences, self.min_count, self.thr)
        # add info about each word's Huffman encoding
        if self.hs:
            self._create_binary_tree()
        # build the table for drawing random words (for negative sampling)
        if self.neg:
            self._make_table()
        # initialize layers
        self.reset_weights()

    def train_sentence_sg(self, sentence, alpha):
        """
        Update a skip-gram model by training on a single sentence (batch mode!)
        using hierarchical softmax and/or negative sampling.

        The sentence is a list of Vocab objects (or None, where the corresponding
        word is not in the vocabulary. Called internally from `Word2Vec.train()`.
        """
        if self.neg:
            # precompute neg noise labels
            labels = np.zeros(self.neg + 1)
            labels[0] = 1.
        for pos, word in enumerate(sentence):
            if not word or (word.prob and word.prob < np.random.rand()):
                continue  # OOV word in the input sentence or subsampling => skip
            reduced_window = np.random.randint(self.window - 1)
            # now go over all words from the (reduced) window (at once), predicting each one in turn
            start = max(0, pos - self.window + reduced_window)
            word2_indices = [word2.index for pos2, word2 in enumerate(
                sentence[start:pos + self.window + 1 - reduced_window], start) if (word2 and not (pos2 == pos))]
            if not word2_indices:
                continue
            l1 = deepcopy(self.wv.vectors[word2_indices])  # len(word2_indices) x layer1_size
            if self.hs:
                # work on the entire tree at once --> 2d matrix, codelen x layer1_size
                l2 = deepcopy(self.syn1[word.point])
                # propagate hidden -> output (len(word2_indices) x codelen)
                f = 1. / (1. + np.exp(-np.dot(l1, l2.T)))
                # vector of error gradients multiplied by the learning rate
                g = (1. - np.tile(word.code, (len(word2_indices), 1)) - f) * alpha
                # learn hidden -> output (codelen x layer1_size) batch update
                self.syn1[word.point] += np.dot(g.T, l1)
                # learn input -> hidden
                self.wv.vectors[word2_indices] += np.dot(g, l2)
            if self.neg:
                # use this word (label = 1) + k other random words not from this sentence (label = 0)
                word_indices = [word.index]
                while len(word_indices) < self.neg + 1:
                    w = self.table[np.random.randint(self.table.shape[0])]
                    if not (w == word.index or w in word2_indices):
                        word_indices.append(w)
                # 2d matrix, k+1 x layer1_size
                l2 = deepcopy(self.syn1neg[word_indices])
                # propagate hidden -> output
                f = 1. / (1. + np.exp(-np.dot(l1, l2.T)))
                # vector of error gradients multiplied by the learning rate
                g = (np.tile(labels, (len(word2_indices), 1)) - f) * alpha
                # learn hidden -> output (batch update)
                self.syn1neg[word_indices] += np.dot(g.T, l1)
                # learn input -> hidden
                self.wv.vectors[word2_indices] += np.dot(g, l2)
        return len([word for word in sentence if word])

    def train_sentence_cbow(self, sentence, alpha):
        """
        Update a cbow model by training on a single sentence
        using hierarchical softmax and/or negative sampling.

        The sentence is a list of Vocab objects (or None, where the corresponding
        word is not in the vocabulary. Called internally from `Word2Vec.train()`.
        """
        if self.neg:
            # precompute neg noise labels
            labels = np.zeros(self.neg + 1)
            labels[0] = 1.
        for pos, word in enumerate(sentence):
            if not word or (word.prob and word.prob < np.random.rand()):
                continue  # OOV word in the input sentence or subsampling => skip
            reduced_window = np.random.randint(self.window - 1)  # how much is SUBSTRACTED from the original window
            # get sum of representation from all words in the (reduced) window (if in vocab and not the `word` itself)
            start = max(0, pos - self.window + reduced_window)
            word2_indices = [word2.index for pos2, word2 in enumerate(
                sentence[start:pos + self.window + 1 - reduced_window], start) if (word2 and not (pos2 == pos))]
            if not word2_indices:
                # in this case the sum would return zeros, the mean nans but really no point in doing anything at all
                continue
            l1 = np.sum(self.wv.vectors[word2_indices], axis=0)  # 1xlayer1_size
            if self.hs:
                # work on the entire tree at once --> 2d matrix, codelen x layer1_size
                l2 = deepcopy(self.syn1[word.point])
                # propagate hidden -> output
                f = 1. / (1. + np.exp(-np.dot(l1, l2.T)))
                # vector of error gradients multiplied by the learning rate
                g = (1. - word.code - f) * alpha
                # learn hidden -> output
                self.syn1[word.point] += np.outer(g, l1)
                # learn input -> hidden, here for all words in the window separately
                self.wv.vectors[word2_indices] += np.dot(g, l2)
            if self.neg:
                # use this word (label = 1) + k other random words not from this sentence (label = 0)
                word_indices = [word.index]
                while len(word_indices) < self.neg + 1:
                    w = self.table[np.random.randint(self.table.shape[0])]
                    if not (w == word.index or w in word2_indices):
                        word_indices.append(w)
                # 2d matrix, k+1 x layer1_size
                l2 = deepcopy(self.syn1neg[word_indices])
                # propagate hidden -> output
                f = 1. / (1. + np.exp(-np.dot(l1, l2.T)))
                # vector of error gradients multiplied by the learning rate
                g = (labels - f) * alpha
                # learn hidden -> output
                self.syn1neg[word_indices] += np.outer(g, l1)
                # learn input -> hidden, here for all words in the window separately
                self.wv.vectors[word2_indices] += np.dot(g, l2)
        return len([word for word in sentence if word])

    def train(self, sentences, alpha=False, min_alpha=False):
        """
        Update the model's embedding and weights from a sequence of sentences (can be a once-only generator stream).
        Each sentence must be a list of strings.
        """
        logger.info("training model on %i vocabulary and %i features" % (len(self.wv.vocab), self.wv.vector_size))
        if not self.wv.vocab:
            self.train_setup(sentences)
        if alpha:
            self.alpha = alpha
        if min_alpha:
            self.min_alpha = min_alpha
        # build the table for drawing random words (for negative sampling)
        # (is usually deleted before saving)
        if self.neg and self.table is None:
            self._make_table()
        start, next_report = time.time(), 20.
        total_words = sum(v.count for v in self.wv.vocab.values())
        word_count = 0
        for sentence_no, sentence in enumerate(sentences):
            # convert input string lists to Vocab objects (or None for OOV words)
            no_oov = [self.wv.vocab.get(word, None) for word in sentence]
            # update the learning rate before every iteration
            alpha = self.min_alpha + (self.alpha - self.min_alpha) * (1. - word_count / total_words)
            # train on the sentence and check how many words did we train on
            # (out-of-vocabulary (unknown) words do not count)
            if self.mtype == 'sg':
                word_count += self.train_sentence_sg(no_oov, alpha)
            elif self.mtype == 'cbow':
                word_count += self.train_sentence_cbow(no_oov, alpha)
            else:
                raise RuntimeError("model type %s not known!" % self.mtype)
            # report progress
            elapsed = time.time() - start
            if elapsed >= next_report:
                logger.info("PROGRESS: at %.2f%% words, alpha %.05f, %.0f words/s" %
                            (100.0 * word_count / total_words, alpha, word_count / elapsed if elapsed else 0.0))
                next_report = elapsed + 20.  # don't flood the log, wait at least a second between progress reports
        elapsed = time.time() - start
        logger.info("training on %i words took %.1fs, %.0f words/s" %
                    (word_count, elapsed, word_count / elapsed if elapsed else 0.0))
        # compute vector norms for later stuff
        self.wv.init_sims()
