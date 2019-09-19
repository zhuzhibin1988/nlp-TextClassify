import jieba  # 分词工具
import jieba.posseg as jb_posseg  # 词性标注
import jieba.analyse as jb_analyse  # 关键词提取

from pyltp import Segmentor  # 分词
from pyltp import Postagger  # 词性标注
from pyltp import Parser  # 依存句法
from pyltp import SementicRoleLabeller  # 角色标注

from sentiment.dependency import Dependency
from sentiment.triple_extraction import TripleExtractor

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer  # 算法工具
import numpy as np  # 矩阵工具
import nltk

import os

"""
1.答案文档分词
2.构造答案词典
3.训练词典单词的tfidf(w)=tf(d,w)*idf(w)
4.计算句子的tfidf vector
5.关键词提取
"""
LTP_MODEL_DIR = "/Users/zhuzhibin/Program/python/qd/nlp/data/ltp_data_v3.4.0"


class CommentParser(object):
    def __init__(self):
        self.segmentor = Segmentor()
        self.postagger = Postagger()
        self.parser = Parser()  # 初始化实例
        self.labeller = SementicRoleLabeller()  # 初始化实例

        self.segmentor.load(os.path.join(LTP_MODEL_DIR, "cws.model"))
        self.postagger.load(os.path.join(LTP_MODEL_DIR, "pos.model"))
        self.parser.load(os.path.join(LTP_MODEL_DIR, "parser.model"))  # 加载模型
        self.labeller.load(os.path.join(LTP_MODEL_DIR, "pisrl.model"))  # 加载模型

    def release(self):
        self.labeller.release()
        self.parser.release()
        self.postagger.release()
        self.segmentor.release()

    def stopwordlist(self, *self_define_stopwords):
        """
        get stopwords list
        :param self_define_stopwords: add self define stop word to stopwords list
        :return: stopwordlist
        """
        stopwords = [word.strip() for word in open("../data/stopwords", "r").readlines()]
        for stopword in self_define_stopwords:
            stopwords.append(stopword)
        return stopwords

    def sentence_segment_ltp(self, comment, print_each=True):
        words = self.segmentor.segment(comment)
        postags = self.postagger.postag(words)

        word_tag_tuple_list = []
        for i in range(len(words)):
            word_tag_tuple_list.append((str(i), words[i], postags[i]))

        arcs = self.parser.parse(words, postags)

        # arcs 使用依存句法分析的结果
        labels = self.labeller.label(words, postags, arcs)  # 语义角色标注
        if print_each:
            print("|".join(words))
            print('  '.join('|'.join(tpl) for tpl in word_tag_tuple_list))
            print("\t".join("%d:%s" % (arc.head, arc.relation) for arc in arcs))
            for label in labels:
                print(label.index,
                      "".join(["%s:(%d,%d)" % (arg.name, arg.range.start, arg.range.end) for arg in label.arguments]))
        print(comment, self.parse_label_opinion(labels, words))
        print(comment, self.parse_pos_opinion(arcs, words))

    def parse_label_opinion(self, labels, words):
        """
        解释role提取观点
        :param labels:
        :param words:
        :return:
        """
        shorts = []
        for label in labels:
            # 谓词索引
            verb_index = label.index
            # 该谓词若干语义角色
            roles = label.arguments

            verb_word = words[verb_index]
            short = ""
            for role in roles:
                short += "".join(words[role.range.start:role.range.end + 1])
            shorts.append(short + verb_word)
        return shorts

    def parse_pos_opinion(self, arcs, words):
        for n, arc in enumerate(arcs):
            if arc.head == 0:
                head = words[n]
                head_index = n + 1
                break

        short = [head]
        for n, arc in enumerate(arcs):
            if arc.head == head_index and arc.relation in [Dependency.SBV.value, Dependency.VOB.value,
                                                           Dependency.ATT.value, Dependency.ADV.value]:
                short.append(words[n])
        return short

    def parse_sentiment(self, arcs):
        for arc in arcs:
            arc.relation

    def parse_head(self, arc, index, words, postags):
        head = words[index]
        return head

    def parse_svb(self, arc, index, words, postags):
        word = words[index]
        head_word = words[arc.head]
        short_sentence = head_word + word
        return short_sentence

    def parse_att(self, arc, index, words, postags):
        word = words[index]
        head_word = words[arc.head]
        short_sentence = head_word + word
        return short_sentence

    def parse_svb(self, arc, index, words, postags):
        word = words[index]
        head_word = words[arc.head]
        short_sentence = head_word + word
        return short_sentence

    def sentence_segment_jieba(self, comment, stopword_list={}):
        """
        预处理，使用jieba对语料分词
        :param comment:
        :param stopword_list:
        :return:
        """
        segment = jieba.lcut(comment.strip())
        if len(stopword_list) > 0:
            temp_segment = []
            for segment_word in segment:
                if segment_word not in stopword_list and segment_word != " ":
                    temp_segment.append(segment_word)
            segment = temp_segment
        # print(comment + ":{0}".format(segment))
        return segment

    def sentence_segment_by_bland(self, comment, stopword_list={}):
        """
        使用空格间隔分词
        如：
        我们 喜欢 吃 冰激凌
        :param comment: 一条语料
        :param stopword_list: 停用词列表
        :return:
        """
        segment = self.sentence_segment_jieba(comment, stopword_list)
        return " ".join(segment)

    def construct_pow(self, use_stopwords):
        """
        构建词袋
        :param use_stopwords:
        :return:
        """
        stopword_list = {}
        if use_stopwords:
            stopword_list = self.stopwordlist("之前", "是因为", "已经")
        dictionary = set()
        with open("../data/comment", "r") as comments:
            for comment in comments:
                segment = self.sentence_segment_jieba(comment, stopword_list)
                for word in segment:
                    dictionary.add(word)
        print(dictionary)

    def train_tfidf(self):
        """
        计算词袋里的词的重要性指标tfidf
        :return:
        """
        corpus = []
        tfidfdict = {}
        with open("../data/comment", "r") as comments:
            for comment in comments:
                corpus.append(self.sentence_segment_by_bland(comment))
        vectorizer = CountVectorizer(stop_words=self.stopwordlist("之前", "是因为", "已经"))
        transformer = TfidfTransformer()
        tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
        word = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
        weight = tfidf.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
        np.set_printoptions(threshold=np.inf)  # 数组不折叠显示
        print(word)
        print(np.array(weight))
        for i in range(len(weight)):  # 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
            for j in range(len(word)):
                getword = word[j]
                getvalue = weight[i][j]
                if getvalue != 0:  # 去掉值为0的项
                    if getword in tfidfdict:  # 更新全局TFIDF值
                        tfidfdict[getword] += float(getvalue)
                    else:
                        tfidfdict.update({getword: float(getvalue)})
        sorted_tfidf = sorted(tfidfdict.items(), key=lambda d: d[1], reverse=True)
        with open("../data/sk_tfidf.txt", 'w') as out:
            for i in sorted_tfidf:  # 写入文件
                out.write(i[0] + '\t' + str(i[1]) + '\n')

    def self_define(self):
        jb_analyse.set_stop_words("../data/stopwords")
        jb_analyse.set_idf_path("../data/sk_tfidf.txt")

    def triple_extract(self, comment):
        extractor = TripleExtractor()
        svos = extractor.triples_main(comment)
        print(svos)


comment = "奶油和蛋糕的配置很合理，不会很腻，奶油的量恰到好处，一层咬下去很好吃，里面的水果也好吃"
# construct_pow(True)
# train_tfidf()

parser = CommentParser()
# with open("../data/comment", "r") as comments:
#     for comment in comments:
#         parser.sentence_segment_ltp(comment, False)

parser.sentence_segment_ltp("第一次买就觉得味道不错，不腻")