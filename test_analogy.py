# import modules and set up logging
from glob import glob
import cPickle as pkl
from copy import deepcopy
import numpy as np
import logging

import word2vec
import context2vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class Text8Corpus(object):
    """Iterate over sentences from the "text8" corpus, unzipped from http://mattmahoney.net/dc/text8.zip ."""

    def __init__(self, fname):
        self.fname = fname

    def __iter__(self):
        # the entire corpus is one gigantic line -- there are no sentence marks at all
        # so just split the sequence of tokens arbitrarily: 1 sentence = 1000 tokens
        sentence, rest, max_sentence_length = [], '', 1000
        with open(self.fname) as fin:
            while True:
                text = rest + fin.read(8192)  # avoid loading the entire file (=1 line) into RAM
                if text == rest:  # EOF
                    sentence.extend(rest.split())  # return the last chunk of words, too (may be shorter/longer)
                    if sentence:
                        yield sentence
                    break
                # the last token may have been split in two... keep it for the next iteration
                last_token = text.rfind(' ')
                words, rest = (text[:last_token].split(), text[last_token:].strip()) if last_token >= 0 else ([], text)
                sentence.extend(words)
                while len(sentence) >= max_sentence_length:
                    yield sentence[:max_sentence_length]
                    sentence = sentence[max_sentence_length:]


class OneBilCorpus(object):
    """Iterate over sentences from the "1-billion-word-language-modeling-benchmark" corpus,
    downloaded from http://code.google.com/p/1-billion-word-language-modeling-benchmark/ ."""

    def __init__(self):
        self.dir = 'data/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/news*'

    def __iter__(self):
        # go file by file
        for fname in glob(self.dir):
            with open(fname) as f:
                yield f.read().lower().split()


def analogy(model, a, b, c):
    # man:woman as king:x - a:b as c:x - find x
    # get embeddings for a, b, and c and multiply with all other words
    a_sims = 1. + np.dot(model.syn0norm, model.syn0norm[model.vocab[a].index])
    b_sims = 1. + np.dot(model.syn0norm, model.syn0norm[model.vocab[b].index])
    c_sims = 1. + np.dot(model.syn0norm, model.syn0norm[model.vocab[c].index])
    # add/multiply them as they should
    return b_sims - a_sims + c_sims
    # return (b_sims*c_sims)/a_sims


def accuracy(model, questions, lowercase=True, restrict_vocab=30000):
    """
    Compute accuracy of the model. `questions` is a filename where lines are
    4-tuples of words, split into sections by ": SECTION NAME" lines.
    See https://code.google.com/p/word2vec/source/browse/trunk/questions-words.txt for an example.

    The accuracy is reported (=printed to log and returned as a list) for each
    section separately, plus there's one aggregate summary at the end.

    Use `restrict_vocab` to ignore all questions containing a word whose frequency
    is not in the top-N most frequent words (default top 30,000).

    This method corresponds to the `compute-accuracy` script of the original C word2vec.

    """
    ok_vocab = dict(sorted(model.vocab.iteritems(), key=lambda item: -item[1].count)[:restrict_vocab])
    ok_index = set(v.index for v in ok_vocab.itervalues())

    def log_accuracy(section):
        correct, incorrect = section['correct'], section['incorrect']
        if correct + incorrect > 0:
            print "%s: %.1f%% (%i/%i)" % (section['section'],
                                          100.0 * correct / (correct + incorrect), correct, correct + incorrect)

    sections, section = [], None
    for line_no, line in enumerate(open(questions)):
        # TODO: use level3 BLAS (=evaluate multiple questions at once), for speed
        if line.startswith(': '):
            # a new section starts => store the old section
            if section:
                sections.append(section)
                log_accuracy(section)
            section = {'section': line.lstrip(': ').strip(), 'correct': 0, 'incorrect': 0}
        else:
            if not section:
                raise ValueError("missing section header before line #%i in %s" % (line_no, questions))
            try:
                if lowercase:
                    a, b, c, expected = [word.lower() for word in line.split()]
                else:
                    a, b, c, expected = [word for word in line.split()]
            except:
                print "skipping invalid line #%i in %s" % (line_no, questions)
            if a not in ok_vocab or b not in ok_vocab or c not in ok_vocab or expected not in ok_vocab:
                # print "skipping line #%i with OOV words: %s" % (line_no, line)
                continue

            ignore = set(model.vocab[v].index for v in [a, b, c])  # indexes of words to ignore
            predicted = None
            # find the most likely prediction, ignoring OOV words and input words
            # for index in np.argsort(model.most_similar(positive=[b, c], negative=[a], topn=False))[::-1]:
            for index in np.argsort(analogy(model, a, b, c))[::-1]:
                if index in ok_index and index not in ignore:
                    predicted = model.index2word[index]
                    # if predicted != expected:
                    #     print "%s: expected %s, predicted %s" % (line.strip(), expected, predicted)
                    break
            section['correct' if predicted == expected else 'incorrect'] += 1
    if section:
        # store the last section, too
        sections.append(section)
        log_accuracy(section)

    total = {'section': 'total', 'correct': sum(s['correct']
                                                for s in sections), 'incorrect': sum(s['incorrect'] for s in sections)}
    log_accuracy(total)
    sections.append(total)
    return sections


def accuracy_examples(model):
    # just as advertised...
    print model.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
    # "boy" is to "father" as "girl" is to ...?
    print model.most_similar(['girl', 'father'], ['boy'], topn=3)
    more_examples = ["he his she", "big bigger bad", "going went being"]
    for example in more_examples:
        a, b, x = example.split()
        predicted = model.most_similar([x, b], [a])[0][0]
        print "'%s' is to '%s' as '%s' is to '%s'" % (a, b, x, predicted)
    # which word doesn't go with the others?
    print model.doesnt_match("breakfast cereal dinner lunch".split())


def evaluate_google():
    # see https://code.google.com/archive/p/word2vec/
    # load pretrained google embeddings and test
    from gensim.models import Word2Vec
    model_google = Word2Vec.load_word2vec_format('data/GoogleNews-vectors-negative300.bin.gz', binary=True)
    _ = accuracy(model_google, "data/questions-words.txt", False)


def evaluate_word2vec(corpus, seed=1):
    # load and evaluate
    fname = "%s_cbow_200_hs0_neg13_seed%i.model" % (corpus, seed)
    with open("data/%s" % fname, 'rb') as f:
        model = pkl.load(f)
    _ = accuracy(model, "data/questions-words.txt")


def evaluate_contextenc(corpus, seed=1):
    # load word2vec model
    print "####### seed = %i" % seed
    fname = "%s_cbow_200_hs0_neg13_seed%i.model" % (corpus, seed)
    with open("data/%s" % fname, 'rb') as f:
        model_org = pkl.load(f)
    # get context matrix
    if corpus == 'text8':
        sentences = Text8Corpus('data/text8')
    elif corpus == '1bil':
        sentences = OneBilCorpus()
    context_model = context2vec.ContextModel(
        sentences, min_count=model_org.min_count, window=model_org.window, wordlist=model_org.index2word)
    for fill_diag in [True, False]:
        model = deepcopy(model_org)
        # build context matrix
        print "constructing context matrix for fill_diag: %s" % (fill_diag)
        context_mat = context_model.get_context_matrix(fill_diag, False)
        # adapt the word2vec model
        print "adapting the word2vec weights - syn0norm"
        model.syn0norm = context_mat.dot(model.syn0norm)
        # renormalize
        model.syn0norm = model.syn0norm / np.array([np.linalg.norm(model.syn0norm, axis=1)]).T
        # evaluate
        print "evaluating the model"
        _ = accuracy(model, "data/questions-words.txt")


def train_word2vec(corpus='text8', seed=1, it=10, save_interm=True):
    # load text
    if corpus == 'text8':
        sentences = Text8Corpus('data/text8')
    elif corpus == '1bil':
        sentences = OneBilCorpus()

    def save_model(model, saven):
        # delete the huge stupid table again
        table = deepcopy(model.table)
        model.table = None
        # pickle the entire model to disk, so we can load&resume training later
        pkl.dump(model, open("data/%s" % saven, 'wb'), -1)
        # reinstate the table to continue training
        model.table = table

    # train the cbow model; default window=5
    model = word2vec.Word2Vec(sentences, mtype='cbow', hs=0, neg=13, embed_dim=200, seed=seed)
    for i in range(1, it):
        print "####### ITERATION %i ########" % i
        _ = accuracy(model, "data/questions-words.txt")
        if save_interm:
            save_model(model, "%s_cbow_200_hs0_neg13_seed%i_it%i.model" % (corpus, seed, i))
        model.train(sentences, alpha=0.005, min_alpha=0.005)
    save_model(model, "%s_cbow_200_hs0_neg13_seed%i_it%i.model" % (corpus, seed, it))
    print "####### ITERATION %i ########" % it
    _ = accuracy(model, "data/questions-words.txt")
    accuracy_examples(model)


def main():
    # load the text on which we're training
    sentences = Text8Corpus('data/text8')
    # train the cbow model; default window=5, min_count=5
    # with open("data/text8_cbow_200_hs0_neg13_seed3.model") as f:
    #     model = pkl.load(f)
    model = word2vec.Word2Vec(sentences, mtype='cbow', hs=0, neg=13, embed_dim=200, seed=3)
    """
        collected 253854 unique words from a corpus of 17005207 words and 17006 sentences
        total of 71290 unique words after removing those with count < 5
        training model on 71290 vocabulary and 200 features
        training on 16718844 words took 2789.4s, 5994 words/s
    """
    # don't need the table used for negative sampling (it's huge)
    model.table = None
    # evaluate the accuracy on the analogy task
    _ = accuracy(model, "data/questions-words.txt")
    """
        capital-common-countries: 13.8% (70/506)
        capital-world: 5.5% (80/1452)
        currency: 2.2% (6/268)
        city-in-state: 12.8% (201/1571)
        family: 58.5% (179/306)
        gram1-adjective-to-adverb: 6.7% (51/756)
        gram2-opposite: 12.1% (37/306)
        gram3-comparative: 41.0% (516/1260)
        gram4-superlative: 26.5% (134/506)
        gram5-present-participle: 13.4% (133/992)
        gram6-nationality-adjective: 34.7% (476/1371)
        gram7-past-tense: 17.7% (236/1332)
        gram8-plural: 28.6% (284/992)
        gram9-plural-verbs: 25.8% (168/650)
        total: 21.0% (2571/12268)
    """
    # get the global context matrix relying on the same text
    context_model = context2vec.ContextModel(sentences, min_count=model.min_count,
                                             window=model.window, wordlist=model.index2word)
    # best results on the analogy task when counting the target word in addition to the context words
    # --> fill diagonal of the context matrix. normalization is irrelevant since we renormalize later
    context_mat = context_model.get_context_matrix(fill_diag=True, norm=False)
    # adapt the word embeddings of the word2vec model by multiplying them with the context matrix
    model.syn0norm = context_mat.dot(model.syn0norm)
    # renormalize so the word embeddings have unit length again
    model.syn0norm = model.syn0norm / np.array([np.linalg.norm(model.syn0norm, axis=1)]).T
    # evaluate the model again
    _ = accuracy(model, "data/questions-words.txt")
    """
        capital-common-countries: 38.3% (194/506)
        capital-world: 19.4% (281/1452)
        currency: 10.8% (29/268)
        city-in-state: 20.9% (328/1571)
        family: 65.7% (201/306)
        gram1-adjective-to-adverb: 9.0% (68/756)
        gram2-opposite: 15.7% (48/306)
        gram3-comparative: 43.3% (546/1260)
        gram4-superlative: 23.5% (119/506)
        gram5-present-participle: 18.8% (186/992)
        gram6-nationality-adjective: 38.3% (525/1371)
        gram7-past-tense: 19.3% (257/1332)
        gram8-plural: 28.7% (285/992)
        gram9-plural-verbs: 21.5% (140/650)
        total: 26.1% (3207/12268)
    """


if __name__ == '__main__':
    main()
