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
from sentiment.pos import Pos
from sentiment.triple_extraction import TripleExtractor

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer  # 算法工具
import numpy as np  # 矩阵工具
import nltk

from gensim import corpora, models

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

        self.adv_dict_list = self.load_adverb_dictionary()
        self.pronoun_list = self.load_pronoun_words()
        self.vi_list = self.load_intransitive_verb()
        self.stopword_list = self.load_stopwords("之前", "是因为", "已经")

    def release(self):
        self.labeller.release()
        self.parser.release()
        self.postagger.release()
        self.segmentor.release()

    @classmethod
    def load_stopwords(cls, *self_define_stopwords):
        """
        get stopwords list
        :param self_define_stopwords: add self define stop word to stopwords list
        :return: stopwords_list
        """
        stopwords_list = [word.strip() for word in open("../data/stopwords.txt", "r").readlines()]
        for stopword in self_define_stopwords:
            stopwords_list.append(stopword)
        return stopwords_list

    @classmethod
    def load_intransitive_verb(cls):
        intransitive_verb = []
        with open("../data/intransitive_verb.txt", "r") as vi_file:
            for word in vi_file.readlines():
                intransitive_verb.append(word.strip())
        return intransitive_verb

    @classmethod
    def load_pronoun_words(cls):
        with open("../data/pronoun.txt", "r") as pronoun_file:
            for line in pronoun_file.readlines():
                pronoun_list = line.split(" ")
        return pronoun_list

    @classmethod
    def load_adverb_dictionary(cls):
        dictionary = {}
        with open("../data/adv.txt", "r") as adv_file:
            for line in adv_file.readlines():
                index = line.index(":")
                key = line[0: index].strip()
                value = line[index + 1:].strip()
                dictionary.update({key: value.split(" ")})
        return dictionary

    def sentence_segment_ltp(self, comment, print_each=True):
        comment = self.format_sentence(comment)
        words = self.segmentor.segment(comment)
        opinions = self.sentence_segment(words, print_each)
        # print(comment, self.distinct_opinion(list(opinions)))
        return opinions

    def all_comment_opinions(self, comments):
        opinions_list = []
        for comment in comments:
            opinions = self.sentence_segment_ltp(comment, False)
            for opinion in opinions[1]:
                opinions_list.append(opinion[2])
        return opinions_list

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
                opi_1 = opinions[m][1]
                opi_2 = opinions[m - 1][1]
                if len(opi_1) > len(opi_2):
                    tmp = opinions[m - 1]
                    opinions[m - 1] = opinions[m]
                    opinions[m] = tmp

        for opinion in opinions:
            opi = opinion[1]
            if len(distinct_opinion_list) == 0:
                distinct_opinion_list.append(opinion)
            else:
                include = False
                for idx in range(0, len(distinct_opinion_list)):
                    try:
                        include |= distinct_opinion_list[idx][1].index(opi) > -1
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
            print("  ".join('|'.join(tpl) for tpl in word_tag_tuple_list))
            print("  ".join("%d|%d:%s" % (n, arc.head, arc.relation) for n, arc in enumerate(arcs)))
            for label in labels:
                print(label.index, "".join(["%s:(%d,%d)" % (arg.name, arg.range.start, arg.range.end) for arg in label.arguments]))
        # print("label", comment, self.parse_label_opinion(labels, words))
        # print("核心观点", comment, self.parse_pos_opinion(arcs, words))

        core_opinion_list = []
        core_opinion = self.parse_core_opinion(arcs, words, postags, core_opinion_list)
        opinions = self.parse_label_opinion(["v", "a", "i", "z"], arcs, words, postags)
        return core_opinion, opinions

    def parse_label_opinion(self, core_pos, arcs, words, postags):
        """
        给出核心词性，解释所有该词性的短语观点
        :param core_pos:
        :param arcs:
        :param words:
        :param postags:
        :return:
        """
        opinions = []
        for n, arc in enumerate(arcs):
            postag = postags[n]
            if postag in [Pos.v.value, Pos.a.value, Pos.i.value] or arc.relation in [Dependency.HED.value, Dependency.COO.value]:
                core_opinion_list = []
                core_opinion_list = self.parse_opinion(n, arcs, words, postags)

                # if postag == "v" and words[n] not in ["有", "想"]:
                #     self.parse_verb_opinion(n, arcs, words, postags, core_opinion_list)
                # elif postag in ["a", "i"]:
                #     self.parse_adj_opinion(n, arcs, words, postags, core_opinion_list)
                # # print(words[n], core_opinion_list)
                if not self.opinion_single_pos(postag, core_opinion_list):
                    opinions.append((postag, words[n], self.opinion_to_str(n, words, core_opinion_list)))
        return opinions

    def parse_core_opinion(self, arcs, words, postags, core_opinion_list=[]):
        for n, arc in enumerate(arcs):
            if arc.relation == Dependency.HED.value:
                core_index = n
        core_pos = postags[core_index]
        return core_pos, words[core_index], self.opinion_to_str(core_index, words, self.parse_opinion(core_index, arcs, words, postags))
        # if core_pos == "v":
        #     self.parse_verb_opinion(core_index, arcs, words, postags, core_opinion_list)
        # elif core_pos == "a":
        #     self.parse_adj_opinion(core_index, arcs, words, postags, core_opinion_list)
        # return core_pos, words[core_index], self.opinion_to_str(core_index, words, core_opinion_list)

    def parse_verb_opinion(self, verb_index, arcs, words, postags, core_opinion_list=[]):
        """

        :param verb_index: 核心动词下标
        :param arcs:
        :param words:
        :param postags:
        :param core_opinion_list: 核心动词的观点词list
        :return:
        """
        has_vob = False
        sbv_word = ()
        sbv_att_words = []
        available_word_idx_list = [verb_index]

        def word_root_index(core_idx, index):
            """
            查找词的root index
            :return:
            """
            arc = arcs[index]
            idx = index if arc.relation == Dependency.HED.value else arc.head - 1
            if idx == core_idx or idx == index:
                return idx
            else:
                return word_root_index(core_idx, idx)

        def verb_opinion(index):
            """
            提取以动词为核心的观点，提取的主要结构主谓结构（SBV）、动宾结构（VOB）、状中结构（ADV）、动补结构（CMP）、介宾结构（POB）
            :return:
            """
            nonlocal has_vob
            nonlocal sbv_word
            nonlocal sbv_att_words
            nonlocal available_word_idx_list

            for m, arc in enumerate(arcs):
                """
                    tuple格式：（index, 句法依存关系, 词性, 词）
                """
                word_tuple = (m, arc.relation, postags[m], words[m])

                """
                    父节点词
                """
                if arc.head == index + 1 and arc.relation in [Dependency.SBV.value, Dependency.VOB.value, Dependency.ATT.value, Dependency.CMP.value,
                                                              Dependency.ADV.value, Dependency.POB.value, Dependency.RAD.value]:
                    """
                        计算词的root词是否等于关键词
                    """
                    root_verb_index = word_root_index(verb_index, m)
                    if root_verb_index == verb_index:
                        if arc.relation == Dependency.VOB.value and (arc.head - 1) == verb_index:
                            has_vob = True
                            available_word_idx_list.append(m)
                            core_opinion_list.append((m, arc.relation, postags[m], words[m]))
                        # elif (arc.relation == Dependency.COO.value and postags[m] != "v") or (arc.relation != Dependency.COO.value and postags[m] not in ["o", "c", "e", "m", "q", "p", "u", "nd", "b"]):
                        elif postags[m] not in ["o", "c", "e", "m", "q", "p", "nd", "b"]:
                            if arc.head - 1 in available_word_idx_list:
                                available_word_idx_list.append(m)
                                """
                                    若是主谓结构先暂存，不加入观点词list
                                """
                                if arc.relation == Dependency.SBV.value:
                                    sbv_word = word_tuple
                                else:
                                    """
                                       计算词的root词是否等于sbv关键词
                                    """
                                    sbv_index = sbv_word[0] if len(sbv_word) > 0 else -1
                                    root_sbv_index = word_root_index(sbv_index, word_tuple[0])
                                    if root_sbv_index == sbv_index:
                                        """
                                            若是主谓结构的其他属性词，暂存在主谓属性词列表
                                        """
                                        sbv_att_words.append(word_tuple)
                                    else:
                                        core_opinion_list.append((m, arc.relation, postags[m], words[m]))
                    verb_opinion(m)

        verb_opinion(verb_index)
        """
        三元组判断，只有包含了动宾结构才把主谓结构加入
        """
        if has_vob and len(sbv_word) > 0:
            core_opinion_list.append(sbv_word)
            core_opinion_list += sbv_att_words

    def parse_adj_opinion(self, adj_index, arcs, words, postags, core_opinion_list=[]):
        """
        提取以形容词为核心的观点，提取的主要结构主谓结构（SBV）、状中结构（ADV）
        :param adj_index:
        :param arcs:
        :param words:
        :param postags:
        :param core_opinion_list:
        :return:
        """
        for m, arc in enumerate(arcs):
            if arc.head == adj_index + 1 and arc.relation in [Dependency.SBV.value, Dependency.ADV.value, Dependency.ATT.value]:
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
        if len(core_opinion_list) == 0 and core_pos == "v":
            return True
        return False

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

    def sentence_segment_by_bland(self, comment, segment_type="ltp", stopword_list={}):
        """
        使用空格间隔分词
        如：
        我们 喜欢 吃 冰激凌
        :param comment: 一条语料
        :param segment_type: 分词工具
        :param stopword_list: 停用词列表
        :return:
        """
        if segment_type == "ltp":
            segment = self.segmentor.segment(comment)
        elif segment_type == "jieba":
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
            stopword_list = self.stopword_list
        dictionary = set()
        with open("../data/comment", "r") as comments:
            for comment in comments:
                segment = self.sentence_segment_jieba(comment, stopword_list)
                for word in segment:
                    dictionary.add(word)
        print(dictionary)

    def train_opinion_tfidf(self, opinions):
        texts = []  # 矩阵
        for opinion in opinions:
            texts.append([word for word in self.sentence_segment_by_bland(opinion).split()])
        dictionary = corpora.Dictionary(texts)
        print(dictionary)
        corpus = [dictionary.doc2bow(text) for text in texts]
        # initialize a model
        tfidf = models.TfidfModel(corpus)
        print(tfidf)
        corpus_tfidf = tfidf[corpus]

        for doc in corpus_tfidf:
            print(doc)

        tfidf_dictionary = {}
        for n, word in enumerate(dictionary):
            pass
        return tfidf_dictionary

    def train_tfidf(self):
        """
        计算词袋里的词的重要性指标tfidf
        :return:
        """
        corpus = []
        tfidfdict = {}
        with open("../data/comment", "r") as comments:
            for comment in comments:
                corpus.append(self.sentence_segment_by_bland())
        vectorizer = CountVectorizer(stop_words=self.load_stopword("之前", "是因为", "已经"))
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

    def smart_split_sentence(self, comment):
        subcomments = re.split(r'[。|！|，|、|？|\.|!|,|\?]', self.format_sentence(comment))
        return subcomments

    def format_sentence(self, comment):
        try:
            index = comment.rfind("因为")
            if index < 0:
                index = comment.rfind("由于")
            comment = comment[index + 2:]
        except ValueError:
            pass
        return re.sub(re.compile(r"(\s+)", re.S), "，", comment.strip())

    def word_self_attention(self, parent_pos, parent_word, current_arc_relation, current_arc_pos, current_word):
        """
            判断词性与依存关系组合的有效性
        :param parent_pos: 父节点的词性
        :param parent_word: 父节点的词
        :param current_arc_relation: 当前节点的依存关系
        :param current_arc_pos: 当前节点的词词性
        :param current_word: 当前节点的词
        :return:
        """
        if parent_pos == Pos.v.value:
            if current_arc_relation == Dependency.SBV.value:
                return True
            if current_arc_relation == Dependency.VOB.value:
                return True
            if current_arc_relation == Dependency.ADV.value and (current_word in self.adv_dict_list.get("肯否副词")
                                                                 or (current_arc_pos == Pos.p.value and current_word in ["由", "用"])):  # 由关晓彤代言
                return True
            if current_arc_relation == Dependency.ATT.value:
                return True
            if current_arc_relation == Dependency.CMP.value:
                return True
        elif parent_pos == Pos.a.value:
            if current_arc_relation == Dependency.SBV.value:  # e.g.:材料新鲜
                return True
            if current_arc_relation == Dependency.ADV.value and (current_word not in self.adv_dict_list.get("程度副词") + self.adv_dict_list.get("范围副词")
                                                                 or (current_arc_pos == Pos.p.value and current_word in ["比"])):  # 比别家好
                return True
            if current_arc_relation == Dependency.ATT.value:
                return True
        elif parent_pos in [Pos.n.value, Pos.nd.value, Pos.ni.value, Pos.nl.value, Pos.ns.value, Pos.nt.value, Pos.nz.value]:
            if current_arc_relation == Dependency.ADV.value:
                return True
            if current_arc_relation == Dependency.ATT.value:  # 属性语义修饰名词
                return True
        elif parent_pos == Pos.p.value:
            if current_arc_relation == Dependency.SBV.value:  # 他给我感觉
                return True
            if current_arc_relation == Dependency.VOB.value:  # 给我感觉
                return True
            if current_arc_relation == Dependency.POB.value:  # 比别家好
                return True
        elif parent_pos in [Pos.i.value, Pos.r.value, Pos.q.value] or current_arc_relation == Dependency.CMP.value:
            return True
        return False

    def parse_opinion(self, core_word_index, arcs, words, postags):
        has_vob = False
        sbv_word = ()
        sbv_att_word_list = []
        available_word_idx_list = [core_word_index]
        opinion_word_list = []

        def word_root_index(core_word_idx, index):
            """
            查找词的root index
            :return:
            """
            arc = arcs[index]
            idx = index if arc.relation == Dependency.HED.value else arc.head - 1
            if idx == core_word_idx or idx == index:
                return idx
            else:
                return word_root_index(core_word_idx, idx)

        def verb_opinion(core_word_idx):
            """
            提取以动词为核心的观点，提取的主要结构主谓结构（SBV）、动宾结构（VOB）、状中结构（ADV）、动补结构（CMP）、介宾结构（POB）
            :return:
            """
            nonlocal has_vob
            nonlocal sbv_word
            nonlocal sbv_att_word_list
            nonlocal available_word_idx_list

            for m, arc in enumerate(arcs):
                """
                    tuple格式：（index, 句法依存关系, 词性, 词）
                """
                current_word_tuple = (m, arc.relation, postags[m], words[m])

                parent_word_index = arc.head - 1
                parent_word_tuple = (parent_word_index, arcs[parent_word_index].relation, postags[parent_word_index], words[parent_word_index])

                if arc.head == core_word_idx + 1 \
                        and (current_word_tuple[2] not in [Pos.wp.value, Pos.o.value, Pos.c.value, Pos.r.value] or (current_word_tuple[2] == Pos.r.value and current_word_tuple[3] not in self.pronoun_list)) \
                        and self.word_self_attention(parent_word_tuple[2], parent_word_tuple[3], current_word_tuple[1], current_word_tuple[2], current_word_tuple[3]):

                    """
                        计算词的root词是否等于关键词
                    """
                    root_core_index = word_root_index(core_word_index, m)
                    if root_core_index == core_word_index:
                        if arc.relation == Dependency.VOB.value or (arc.relation == Dependency.CMP.value and postags[current_word_tuple[0]] == Pos.a.value):
                            has_vob = True
                            available_word_idx_list.append(m)
                            opinion_word_list.append(current_word_tuple)
                        else:
                            if arc.head - 1 in available_word_idx_list:
                                available_word_idx_list.append(m)
                                """
                                    若是主谓结构先暂存，不加入观点词list
                                """
                                if arc.relation == Dependency.SBV.value:
                                    if len(sbv_word) == 0:
                                        sbv_word = current_word_tuple
                                else:
                                    """
                                       计算词的root词是否等于sbv关键词
                                    """
                                    sbv_index = sbv_word[0] if len(sbv_word) > 0 else -1
                                    root_sbv_index = word_root_index(sbv_index, current_word_tuple[0])
                                    if root_sbv_index == sbv_index:
                                        """
                                            若是主谓结构的其他属性词，暂存在主谓属性词列表
                                        """
                                        sbv_att_word_list.append(current_word_tuple)
                                    else:
                                        opinion_word_list.append(current_word_tuple)
                        verb_opinion(m)

        verb_opinion(core_word_index)
        """
        三元组判断，只有包含了动宾结构才把主谓结构加入
        """
        if (has_vob or postags[core_word_index] == Pos.a.value or words[core_word_index] in self.vi_list) \
                and len(sbv_word) > 0:
            opinion_word_list.append(sbv_word)
            opinion_word_list += sbv_att_word_list

        return opinion_word_list


comment = "奶油和蛋糕的配置很合理，不会很腻，奶油的量恰到好处，一层咬下去很好吃，里面的水果也好吃"
# construct_pow(True)
# train_tfidf()

parser = CommentParser()
# with open("../data/comment2", "r", encoding="utf-8") as comments:
#     # for comment in comments:
#     #     opinions = parser.sentence_segment_ltp(comment)
#     #     print(comment, "main:", opinions[0], "others:", opinions[1], "\n")
#
#     opinions = parser.all_comment_opinions(comments)
# print(opinions)
# print(parser.train_opinion_tfidf(opinions))

subcomments = parser.smart_split_sentence("第一点是关晓彤是代言人，它本身就是一个具有话题的一个明星，第二，我觉得它会流行起来，它很方便，比较受学生欢迎，第三，它的外观做得也挺不错的")
print(subcomments)
for subcomment in subcomments:
    opinions = parser.sentence_segment_ltp(subcomment)
    print(subcomment, "main:", opinions[0], "others:", opinions[1], "\n")

# parser.sentence_segment_ltp("用当红明星推荐，并且商品很好看")
# parser.sentence_segment_ltp("做活动  买了几个  吃起来味道超级好  做活动还能保证口感  已经很厉害了")
# parser.sentence_segment_ltp("最好不要再加上保护肾脏的")
# parser.sentence_segment_ltp("吃一口觉得味蕾被迷住了")
# parser.sentence_segment_ltp("第一次吃的时候，感觉不会特别甜，里面的蛋糕也很软，有淡淡的甜味 ")
# parser.sentence_segment_ltp("做活动  买了几个  吃起来味道超级好  做活动还能保证口感  已经很厉害了")
# parser.sentence_segment_ltp("朋友推荐，自己尝试，吃了不腻")
# parser.sentence_segment_ltp("购买过一次四重奏，四种口味都非常喜欢。以后就喜欢上了幸福西饼的蛋糕")
# parser.sentence_segment_ltp("对幸福西饼的蛋糕口感印象深刻 ")
# parser.sentence_segment_ltp("奶油和蛋糕的配置很合理，不会很腻，奶油的量恰到好处，一层咬下去很好吃，里面的水果也好吃 ")
# parser.sentence_segment_ltp("产品颜值高，多口味，更健康")
# parser.sentence_segment_ltp("之前吃了一个貌似是榴莲千层雪的蛋糕，口感超棒！")
# parser.sentence_segment_ltp("蛋糕蓬松，奶油味道香 ")
# parser.sentence_segment_ltp("第一次吃的时候，感觉不会特别甜，里面的蛋糕也很软，有淡淡的甜味 ")
# parser.sentence_segment_ltp("入口即化，软绵绵的感觉，吃了还想吃，根本停不下来")
# parser.sentence_segment_ltp("芒果蛋糕口味比别家好太多了 ")