import gensim
from gensim.models import KeyedVectors
from gensim.models import doc2vec

from gensim import models

import distance  # 编辑距离
import numpy as np
from gensim.models.doc2vec import TaggedDocument, Doc2Vec

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sentiment.opinion_extract import OpinionExtractor

WORD_2_VEC_PATH = "/Users/zhuzhibin/Program/python/qd/nlp/data/词向量/sgns.baidubaike.bigram-char"


# WORD_2_VEC_PATH = "/Users/zhuzhibin/Program/python/qd/nlp/data/词向量/Tencent_AILab_ChineseEmbedding.txt"


class OpinionSimilar(object):
    """
    观点相似度
    """

    def __init__(self, embedding):
        self.opinion_extract = OpinionExtractor()
        self.embedding = embedding

    @classmethod
    def load_embedding(cls):
        print("开始加载词向量")
        embedding = KeyedVectors.load_word2vec_format(WORD_2_VEC_PATH, binary=False)  # 使用预训练的词向量
        print("加载词向量完成")
        return embedding

    def check_content_similar(self, content1, content2, tfidf=None):
        """
        计算短句相似度
        :param content1:
        :param content2:
        :param embedding:
        :param tfidf:
        :return:
        """
        dist = self.__cosin_distance(self.sentence_to_vector(content1, self.embedding, tfidf), self.sentence_to_vector(content2, self.embedding, tfidf))
        return dist

    @classmethod
    def cosin_distance(cls, vector1, vector2):
        """
        计算两个向量的余弦距离
        :param vector1:
        :param vector2:
        :return:
        """
        cos_distance = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * (np.linalg.norm(vector2)))
        return cos_distance

    def sentence_to_vector(self, sentence, embedding, tfidf):
        """
        句子to向量
        :param sentence:
        :param embedding:
        :param tfidf:
        :return:
        """
        words, sentence_with_space = self.opinion_extract.sentence_segment_add_space(sentence)
        sentence_vector = np.zeros(300)
        for word in words:
            if tfidf is None:
                sentence_vector += embedding[word]
            else:
                sentence_vector += embedding[word] * tfidf[word]
        sentence_vector = sentence_vector / len(words)
        return sentence_vector

    def train_tf(self, corpora):
        """
        训练词频tf
        :param corpora:
        :return:
        """

        def softmax():
            z = np.array([item[1] for item in dict.items()])
            for word in dict.keys():
                word_softmax = np.exp(dict[word]) / sum(np.exp(z))
                dict.update({word: word_softmax})

        corpus_list = []
        for corpus in corpora:
            words, sentence_with_space = self.opinion_extract.sentence_segment_add_space(corpus)
            corpus_list.append(sentence_with_space)

        tfidf_model = TfidfVectorizer(analyzer='word', token_pattern=r"(?u)\b\w+\b")
        tfidf = tfidf_model.fit_transform(corpus_list)
        cbow = tfidf_model.get_feature_names()
        tf_matrix = tfidf.toarray()
        dict = {}
        for i in range(len(tf_matrix)):
            for j, word in enumerate(cbow):
                e_v = tf_matrix[i][j]
                if e_v != 0:  # 去掉值为0的项
                    if word in dict:  # 更新全局TFIDF值
                        dict[word] += float(e_v)
                    else:
                        dict.update({word: float(e_v)})
        softmax()
        return cbow, dict

    def create_embedding(self, cbow):

        # def get_coefs(word, *arr):
        #     return word, np.asarray(arr, dtype='float32')

        max_features = len(cbow)
        embeddings_index = dict((word, self.embedding[word]) for word in cbow if word in self.embedding.vocab.keys())
        all_embs = np.stack(list(embeddings_index.values()))
        emb_mean, emb_std = all_embs.mean(), all_embs.std()
        embed_size = all_embs.shape[1]
        embedding_matrix = np.random.normal(emb_mean, emb_std, (max_features, embed_size))  # 使用正太分布为未登录词生成随机向量
        for i, word in enumerate(cbow):
            if not embeddings_index.__contains__(word):
                embeddings_index.update({word: embedding_matrix[i]})
        # np.set_printoptions(threshold=np.inf)
        # print(embeddings_index)
        return embeddings_index


def train(corpus):
    # documents = [TaggedDocument(parser.sentence_segment_by_bland(doc), [doc]) for n, doc in enumerate(opinions)]
    # print(documents)
    # model = Doc2Vec(documents, vector_size=100, window=4, min_count=5, workers=4, alpha=0.025, min_alpha=0.0005) 
    pass


def opinion_cluster(opinions):
    opinions.sort(key=lambda opinion: len(opinion), reverse=False)
    opinion_similar = OpinionSimilar()
    cbow, tfidf = opinion_similar.train_tf(opinions)
    embedding = opinion_similar.create_embedding(cbow)
    opinion_classify_dict = {}
    for opinion in opinions:
        classify = False
        if len(opinion_classify_dict) == 0:
            opinion_classify_dict.update({1: [opinion]})
        else:
            for type_category, type_opinions in opinion_classify_dict.items():
                type_opinion = type_opinions[0]
                similar_degree = opinion_similar.check_content_similar(opinion, type_opinion, embedding, tfidf)
                if similar_degree > 0.7:
                    type_opinions.append(opinion)
                    classify = True
                    break
            if not classify:
                new_type_category = type_category + 1
                opinion_classify_dict.update({new_type_category: [opinion]})
    return opinion_classify_dict


def test():
    d = dict({"a": [1, 1, 1, 1], "b": [2, 2, 2, 2]})
    a = np.array([1, 1, 1])
    b = np.array([2, 2, 2])
    c = np.stack(d.values())
    print(a * 0.8)
    print(c.shape[1])


def main():
    opinion_extractor = OpinionExtractor()
    # corpora = []
    # with open("../data/comment", "r", encoding="utf-8") as comments:
    #     for comment in comments:
    #         corpora += [opinion for opinion in opinion_extractor.extract_opinion(comment.strip())]
    # print(corpora)

    # opinions = []
    # with open("../data/comment", "r", encoding="utf-8") as comments:
    #     for comment in comments:
    #         opinions += opinion_extractor.extract_opinion(comment)
    # opinion_classify_dict = opinion_cluster(opinions)
    #
    # for item in opinion_classify_dict.items():
    #     print("category", item[0], ":", item[1])

    opinion_similar = OpinionSimilar()
    print(opinion_similar.check_content_similar("京东", "淘宝"))
    # print(opinion_similar.check_content_similar("包装清新", "觉得味可滋包装很差"))


if __name__ == '__main__':
    main()