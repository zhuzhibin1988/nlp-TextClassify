import operator
import re
import time

import spacy
import neuralcoref
import textacy
from gensim import similarities
from gensim.models import KeyedVectors
from nltk import Tree
from spacy import displacy
from spacy.lang.en import English
from spacy.tokens.token import Token
from spacy.tokenizer import Tokenizer
import numpy as np

from similar.opinion_similar import OpinionSimilar

nlp = spacy.load('en_core_web_sm')  # 加载预训练模型


def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
    else:
        return node.orth_


def get_spacy_word_vector(word):
    word_id = nlp.vocab.strings[word]
    return nlp.vocab.vectors[word_id]


# start = time.time()
# embedding = KeyedVectors.load_word2vec_format("/Users/zhuzhibin/Program/python/qd/nlp/nlp-platform/opinion-data/embedding/GoogleNews-vectors-negative300.bin", binary=True)
# end = time.time()
# print("load cost: %s s" % (end - start))
# # print(embedding["apple"])
#
#
# print(OpinionSimilar.cosin_distance(get_spacy_word_vector("good"), get_spacy_word_vector("well")))
# print(OpinionSimilar.cosin_distance(embedding["good"], embedding["well"]))


prefix_re = spacy.util.compile_prefix_regex(English.Defaults.prefixes)
suffix_re = spacy.util.compile_suffix_regex(English.Defaults.suffixes)
infix_re = spacy.util.compile_infix_regex(English.Defaults.infixes)
pattern_re = re.compile(r'[w*-w*]*')
tokenizer = Tokenizer(nlp.vocab,
    English.Defaults.tokenizer_exceptions,
    prefix_re.search,
    suffix_re.search,
    infix_re.finditer,
    token_match=pattern_re.match)
nlp.tokenizer = tokenizer


# displacy.serve(doc, style='dep')
# [to_nltk_tree(sent.root).pretty_print()
#     for sent in doc.sents]
# for sent in doc.sents:
#     print(sent)
# print('#' * 50)

# token: Token
# for token in doc:
#     print(token, token.pos_, token.lemma_, token.n_lefts, token.n_rights)

# 提取名词短语
# noun_chunks = [nc for nc in doc.noun_chunks]


# print(noun_chunks)

# 知识提取
# statements = textacy.extract.semistructured_statements(doc, "baby clothing")
# for statement in statements:
#     subject, verb, fact = statement
# print("f- ", fact)


def neuralcoref(corpus):
    # 指代消除
    neuralcoref.add_to_pipe(nlp)
    doc = nlp(corpus)
    print(doc._.coref_clusters)


def textrank(corpus):
    import pytextrank
    tr = pytextrank.TextRank()
    nlp.add_pipe(tr.PipelineComponent, name="textrank", last=True)
    doc = nlp(corpus)
    print("=" * 50)
    for phrase in doc._.phrases:
        print(phrase)
        print("{:.4f} {:5d}  {}".format(phrase.rank, phrase.count, phrase.text))
        print(phrase.chunks)
    print("=" * 50)
    for sent in doc._.textrank.summary(limit_phrases=15, limit_sentences=5):
        print(sent)


def gemsim_tfidf(corpus):
    from gensim.models import TfidfModel
    from gensim.corpora import Dictionary
    corpus = [re.sub(r'[.|,]', '', line.lower()).split() for line in corpus]
    dct = Dictionary(corpus)
    corpus_as_bow = [dct.doc2bow(line) for line in corpus]
    print(corpus)
    sort_list = []
    for key in dct.keys():
        sort_list.append((key, dct[key], dct.dfs[key] * dct.cfs[key]))

    sort_list = sorted(sort_list, key=lambda item: item[2], reverse=True)
    keywords_list = sort_list[0:2 if len(sort_list) > 2 else len(sort_list)]

    keywords_doc = [word[1] for word in keywords_list]
    print(keywords_doc)

    print(dct.token2id)
    model_trained = TfidfModel(corpus_as_bow)
    for doc in model_trained[corpus_as_bow]:
        print(doc)

    index = similarities.MatrixSimilarity(model_trained[corpus_as_bow])
    keywords_tfidf = model_trained[dct.doc2bow(keywords_doc)]
    print(keywords_tfidf)
    sims = index[keywords_tfidf]
    print(sims)
    print(max(sims))
    max_idx = np.argmax(sims)
    print("most similar doc is {}".format(corpus[max_idx]))


def load_google_vector():
    nlp.vocab.reset_vectors(width=300)
    embedding = KeyedVectors.load_word2vec_format("/Users/zhuzhibin/Program/python/qd/nlp/nlp-platform/opinion-data/embedding/GoogleNews-vectors-negative300.bin", binary=True)
    for word in embedding.vocab:
        vector = embedding[word]
        nlp.vocab.set_vector(word, vector)
    print("load embedding finish")

    docs = [nlp(u"dog bites man"), nlp(u"man bites dog"),
            nlp(u"man dog bites"), nlp(u"dog man bites")]

    for doc in docs:
        for other_doc in docs:
            print(doc.similarity(other_doc))


if __name__ == '__main__':
    corpus = ["unique painting",
              "beef noodle",
              "unique",
              "visual enjoyment",
              "visual enjoyment"]
    gemsim_tfidf(corpus)
    # load_google_vector()