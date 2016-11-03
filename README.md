With this code you can train and evaluate Context Encoders (ConEc), an extension of word2vec, which can learn word embeddings from large corpora and create out-of-vocabulary embeddings on the spot as well as distinguish between multiple meanings of words based on their local contexts. For further information see: http://openreview.net/forum?id=SkBsEQYll


dependencies: (main code) numpy, scipy; (experiments) sklearn, unidecode

To run the analogy experiment, it is assumed that the [text8 corpus](http://mattmahoney.net/dc/text8.zip) as well as the [analogy questions](https://code.google.com/archive/p/word2vec/) are in a data directory.

To run the named entity recognition experiment, it is assumed that the corresponding [training and test files](http://www.cnts.ua.ac.be/conll2003/ner/) are located in the data/conll2003 directory.


If you have any questions please don't hesitate to send me an [email](mailto:cod3licious@gmail.com) and of course if you should find any bugs or want to contribute other improvements, pull requests are very welcome!
