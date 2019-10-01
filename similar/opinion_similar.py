import gensim
from gensim.models import KeyedVectors
from gensim.models import doc2vec

from gensim import models

import distance  # 编辑距离
import numpy as np
from gensim.models.doc2vec import TaggedDocument, Doc2Vec

from sentiment.opinion_extract import OpinionExtractor

WORD_2_VEC_PATH = "/Users/zhuzhibin/Program/python/qd/nlp/data/词向量/sgns.baidubaike.bigram-char"


# WORD_2_VEC_PATH = "/Users/zhuzhibin/Program/python/qd/nlp/data/词向量/Tencent_AILab_ChineseEmbedding.txt"


class OpinionSimilar(object):
    """
    观点相似度
    """
    def __init__(self):
        self.opinion_extract = OpinionExtractor()

        self.word2vector = self.__load_word_vector()

    @classmethod
    def __load_word_vector(cls):
        print("开始加载词向量")
        word2vector = KeyedVectors.load_word2vec_format(WORD_2_VEC_PATH, binary=False)  # 使用预训练的词向量
        print("加载词向量完成")
        return word2vector

    def check_content_similar(self, content1, content2):
        """
        计算短句相似度
        :param content1:
        :param content2:
        :return:
        """
        dist = self.__cosin_distance(self.sentence_to_vector(content1), self.sentence_to_vector(content2))
        return dist

    @classmethod
    def __cosin_distance(cls, vector1, vector2):
        """
        计算两个向量的余弦距离
        :param vector1:
        :param vector2:
        :return:
        """
        cos_distance = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * (np.linalg.norm(vector2)))
        return cos_distance

    def __word_to_vector(self, word):
        if word in self.word2vector.wv.vocab.keys():
            return self.word2vector[word]
        else:
            return np.zeros(300, dtype='float32')

    def sentence_to_vector(self, sentence):
        words, sentence_with_space = self.opinion_extract.sentence_segment_add_space(sentence)
        sentence_vector = np.zeros(300)
        for word in words:
            sentence_vector += self.__word_to_vector(word)
        sentence_vector = sentence_vector / len(words)
        return sentence_vector

    # def create_random_word_vector(self):
    #     EMBEDDING_FILE = embedding_file
    #     max_features = len(word_index)
    #     print(max_features)
    #
    #     def get_coefs(word, *arr):
    #         return word, np.asarray(arr, dtype='float32')
    #
    #     embeddings_index = dict(get_coefs(*o.strip().split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if
    #                             len(o.strip().split(" ")) == 301 and o.split(" ")[0] in word_index)
    #     all_embs = np.stack(embeddings_index.values())
    #     emb_mean, emb_std = all_embs.mean(), all_embs.std()
    #     embed_size = all_embs.shape[1]
    #     print(len(embeddings_index))
    #     embedding_matrix = np.random.normal(emb_mean, emb_std, (max_features, embed_size))
    #     for word, i in word_index.items():
    #         if i >= max_features:
    #             continue
    #         embedding_vector = embeddings_index.get(word)
    #         if embedding_vector is not None:
    #             embedding_matrix[i] = embedding_vector
    #     np.save(outfile, embedding_matrix)


def train():
    # documents = [TaggedDocument(parser.sentence_segment_by_bland(doc), [doc]) for n, doc in enumerate(opinions)]
    # print(documents)
    # model = Doc2Vec(documents, vector_size=100, window=4, min_count=5, workers=4, alpha=0.025, min_alpha=0.0005) 
    pass
                                                                                                                        

def opinion_classify(opinions):                                                                      
    opinion_similar = OpinionSimilar()
    opinion_classify_dict = {}
    for opinion in opinions:
        classify = False
        if len(opinion_classify_dict) == 0:
            opinion_classify_dict.update({1: [opinion]})
        else:
            for type_category, type_opinions in opinion_classify_dict.items():
                type_opinion = type_opinions[0]
                similar_degree = opinion_similar.check_content_similar(opinion, type_opinion)
                if similar_degree > 0.75:
                    type_opinions.append(opinion)
                    classify = True
            if not classify:
                new_type_category = type_category + 1
                opinion_classify_dict.update({new_type_category: [opinion]})
    return opinion_classify_dict


def main():
    opinion_extractor = OpinionExtractor()
    opinions = []
    # with open("../data/comment", "r", encoding="utf-8") as comments:
    #     for comment in comments:
    #         opinions += opinion_extractor.extract_opinion(comment)
    # print(opinion_classify(opinions))

    opinion_similar = OpinionSimilar()
    print(opinion_similar.check_content_similar("榴莲味浓郁", "包装颜值高"))
    # print(opinion_similar.check_content_similar("包装清新", "觉得味可滋包装很差"))


if __name__ == '__main__':
    main()