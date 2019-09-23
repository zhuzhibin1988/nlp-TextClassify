import jieba  # 分词工具
import jieba.posseg as jb_posseg  # 词性标注
import jieba.analyse as jb_analyse  # 关键词提取

from pyltp import Segmentor  # 分词
from pyltp import Postagger  # 词性标注
from pyltp import Parser  # 依存句法
from pyltp import SementicRoleLabeller  # 角色标注
from pyltp import SentenceSplitter  # 分句

import re

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
        comment = self.format_sentence(comment)
        words_list = [self.segmentor.segment(comment)]
        opinions = set()
        for words in words_list:
            opinions.update(self.sentence_segment(words, print_each))
        # print(comment, self.distinct_opinion(list(opinions)))
        print(comment, opinions)

    @classmethod
    def distinct_opinion(cls, opinions):
        """
        观点去重
        :param opinions:
        :return:
        """
        distinct_opinion_list = []
        for n in range(1, len(opinions)):
            for m in range(n, 0, -1):
                if len(opinions[m]) > len(opinions[m - 1]):
                    tmp = opinions[m - 1]
                    opinions[m - 1] = opinions[m]
                    opinions[m] = tmp

        for opinion in opinions:
            if len(distinct_opinion_list) == 0:
                distinct_opinion_list.append(opinion)
            else:
                include = False
                for idx in range(0, len(distinct_opinion_list)):
                    try:
                        include |= distinct_opinion_list[idx].index(opinion) > -1
                    except ValueError:
                        pass
                if not include:
                    distinct_opinion_list.append(opinion)

        return distinct_opinion_list

    def sentence_segment(self, words, print_each=True):
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
                print(label.index, "".join(["%s:(%d,%d)" % (arg.name, arg.range.start, arg.range.end) for arg in label.arguments]))
        # print("label", comment, self.parse_label_opinion(labels, words))
        # print("核心观点", comment, self.parse_pos_opinion(arcs, words))

        result = self.parse_label_opinion2(["v", "a", "i"], arcs, words, postags)
        return result

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

    def parse_label_opinion2(self, core_pos, arcs, words, postags):
        """
        给出核心词性，解释所有该词性的短语观点
        :param core_pos:
        :param arcs:
        :param words:
        :param postags:
        :return:
        """
        opinions = []
        for n, postag in enumerate(postags):
            if postag in core_pos:
                core_opinion_list = []
                if postag == "v":
                    self.parse_verb_opinion(n, arcs, words, postags, core_opinion_list)
                elif postag == "a":
                    self.parse_adj_opinion(n, arcs, words, postags, core_opinion_list)
                # print(words[n], core_opinion_list)
                if len(core_opinion_list) > 0:
                    opinions.append((postag, words[n], self.opinion_to_str(n, words, core_opinion_list)))
        return opinions

    def parse_verb_opinion(self, core_index, arcs, words, postags, core_opinion_list=[]):
        sbv_att_words = []
        sbv_word = ()
        has_vob = False
        available_arc_head_list = [core_index + 1]
        root_index = None

        def word_root_index(index):
            """
            查找词的root index
            :return:
            """
            arc = arcs[index]
            idx = index if arc.relation == Dependency.HED.value else arc.head - 1
            if idx == core_index or idx == index:
                nonlocal root_index
                root_index = idx
            else:
                word_root_index(idx)

        def verb_opinion(index):
            """
            提取以动词为核心的观点，提取的主要结构主谓结构（SBV）、动宾结构（VOB）、状中结构（ADV）、动补结构（CMP）、介宾结构（POB）
            :return:
            """
            nonlocal sbv_att_words
            nonlocal sbv_word
            nonlocal has_vob
            nonlocal available_arc_head_list

            for m, arc in enumerate(arcs):
                if arc.head == index + 1 and arc.relation in [Dependency.SBV.value, Dependency.VOB.value, Dependency.ATT.value, Dependency.CMP.value, Dependency.ADV.value, Dependency.POB.value, Dependency.RAD.value, Dependency.COO.value]:
                    word_tuple = (m, arc.relation, postags[m], words[m])
                    if arc.relation == Dependency.VOB.value:
                        has_vob = True
                    """
                    词性过滤
                    """
                    if (arc.relation == Dependency.COO.value and postags[m] == "d") or (arc.relation not in [Dependency.HED.value, Dependency.COO.value] and postags[m] not in ["o", "c", "e", "m", "q", "p", "u", "nd", "b"]):
                        """
                            tuple格式：（index, 句法依存关系, 词性, 词）
                        """
                        word_root_index(m)
                        if root_index == core_index:
                            available_arc_head_list.append(m + 1)
                            if arc.head in available_arc_head_list:
                                if arc.relation == Dependency.SBV.value:
                                    sbv_word = word_tuple
                                    word_root_index(sbv_word[0])
                                else:
                                    if len(sbv_word) > 0 and root_index == sbv_word[0] and arc.relation != Dependency.SBV.value:
                                        sbv_att_words.append(word_tuple)
                                    else:
                                        core_opinion_list.append((m, arc.relation, postags[m], words[m]))
                    idx = m + 1 if arc.relation == Dependency.HED.value else m
                    verb_opinion(idx)

        verb_opinion(core_index)
        """
        三元组判断，只有包含了动宾结构才把主谓结构加入
        """
        if has_vob and len(sbv_word) > 0:
            core_opinion_list.append(sbv_word)
            core_opinion_list += sbv_att_words

    def parse_adj_opinion(self, core_index, arcs, words, postags, core_opinion_list=[]):
        """
        提取以形容词为核心的观点，提取的主要结构主谓结构（SBV）、状中结构（ADV）
        :param index:
        :param arcs:
        :param words:
        :param postags:
        :param core_opinion_list:
        :return:
        """
        for m, arc in enumerate(arcs):
            if arc.head == core_index + 1 and arc.relation in [Dependency.SBV.value, Dependency.ADV.value, Dependency.ATT.value]:
                """
                词性过滤
                """
                if arc.relation != Dependency.HED.value and postags[m] not in ["o", "c", "e", "p", "u", "nd"]:
                    core_opinion_list.append((m, arc.relation, postags[m], words[m]))
                idx = m + 1 if arc.relation == Dependency.HED.value else m
                self.parse_adj_opinion(idx, arcs, words, postags, core_opinion_list)

    @classmethod
    def opinion_single_pos(cls, core_pos, core_opinion_list):
        """
        是否只有一种词性
        :param core_pos:
        :param core_opinion_list:
        :return:
        """
        if len(core_opinion_list) == 0:
            return True
        single = False
        for opinion in core_opinion_list:
            single &= core_pos != opinion[2] or (core_pos == opinion[2] and Dependency.VOB.value == opinion[1])
        return single

    @classmethod
    def opinion_to_str(cls, core_word_index, words, core_opinion_list):
        """
        输出观点字符串
        :param core_word_index:
        :param words:
        :param core_opinion_list:
        :return:
        """
        index_list = [core_word_index]
        for core_opinion in core_opinion_list:
            index = core_opinion[0]
            index_list.append(index)
        index_list.sort()

        opinion = ""
        for index in index_list:
            opinion += words[index]
        return opinion

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

    def smart_split_sentence(self, comment):
        sentences = re.split('(。|！|，|、|？|\.|!|,|\?)', comment)
        for sentence in sentences:
            words = self.segmentor.segment(comment)
            postags = self.postagger.postag(words)

    def format_sentence(self, comment):
        try:
            index = comment.index("因为")
            if index < 0:
                index = comment.index("由于")
            comment = comment[index + 2:]
        except ValueError:
            pass
        return re.sub(re.compile(r"(\s+)", re.S), "，", comment.strip())


comment = "奶油和蛋糕的配置很合理，不会很腻，奶油的量恰到好处，一层咬下去很好吃，里面的水果也好吃"
# construct_pow(True)
# train_tfidf()

parser = CommentParser()
# with open("../data/comment", "r") as comments:
#     for comment in comments:
#         parser.sentence_segment_ltp(comment, False)

# parser.sentence_segment_ltp("不腻软罗糯")
# parser.sentence_segment_ltp("做活动  买了几个  吃起来味道超级好  做活动还能保证口感  已经很厉害了")
# parser.sentence_segment_ltp("最好不要再加上保护肾脏的")
# parser.sentence_segment_ltp("吃一口觉得味蕾被迷住了")
# parser.sentence_segment_ltp("第一次吃的时候，感觉不会特别甜，里面的蛋糕也很软，有淡淡的甜味 ")
# parser.sentence_segment_ltp("做活动  买了几个  吃起来味道超级好  做活动还能保证口感  已经很厉害了")
# parser.sentence_segment_ltp("朋友推荐，自己尝试，吃了不腻")
# parser.sentence_segment_ltp("购买过一次四重奏，四种口味都非常喜欢。以后就喜欢上了幸福西饼的蛋糕")
# parser.sentence_segment_ltp("爱上四重奏了，家人每隔一段时间就要吃一个哈哈哈哈哈哈都赞不绝口")
# parser.sentence_segment_ltp("奶油和蛋糕的配置很合理，不会很腻，奶油的量恰到好处，一层咬下去很好吃，里面的水果也好吃 ")
# parser.sentence_segment_ltp("产品颜值高，多口味，更健康")
# parser.sentence_segment_ltp("根本停不下来")
# parser.sentence_segment_ltp("看上去不太好闻")  # parser.sentence_segment_ltp("去屑效果好")
parser.sentence_segment_ltp("家人朋友吃了都说好吃噢，不是很腻，喜欢吃")