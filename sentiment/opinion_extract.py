import logging
from pyltp import Segmentor  # 分词
from pyltp import Postagger  # 词性标注
from pyltp import Parser  # 依存句法
from pyltp import SementicRoleLabeller  # 角色标注
from pyltp import SentenceSplitter  # 分句

import time

from sentiment.dependency import Dependency
from sentiment.pos import Pos

import re
import os

LTP_MODEL_DIR = "/Users/zhuzhibin/Program/python/qd/nlp/data/ltp_data_v3.4.0"

DICTIONARY_DIR = "../data/dictionary"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

rq = time.strftime('%Y%m%d', time.localtime(time.time()))
log_path = os.path.dirname(os.getcwd()) + '/logs/'
log_name = log_path + rq + '.log'
logfile = log_name
# 输出到文件
fh = logging.FileHandler(logfile, mode='a')
fh.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
fh.setFormatter(formatter)
logger.addHandler(fh)
# 输出到控制台
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)  # 输出到console的log等级的开关
ch.setFormatter(formatter)
logger.addHandler(ch)


class OpinionExtractor(object):
    def __init__(self):
        self.__segmentor = Segmentor()
        self.__postagger = Postagger()
        self.__parser = Parser()  # 初始化实例
        self.__labeller = SementicRoleLabeller()  # 初始化实例

        self.__segmentor.load_with_lexicon(os.path.join(LTP_MODEL_DIR, "cws.model"), os.path.join(DICTIONARY_DIR, "custom_lexicon.model"))
        self.__postagger.load(os.path.join(LTP_MODEL_DIR, "pos.model"))
        self.__parser.load(os.path.join(LTP_MODEL_DIR, "parser.model"))  # 加载模型
        self.__labeller.load(os.path.join(LTP_MODEL_DIR, "pisrl.model"))  # 加载模型

        self.__adv_dict_list = self.__load_adverb_dictionary()
        self.__adv_list = self.__adv_dict_list.get("范围副词") + self.__adv_dict_list.get("频率副词") \
                          + self.__adv_dict_list.get("程度副词") + self.__adv_dict_list.get("时间副词") \
                          + self.__adv_dict_list.get("肯否副词") + self.__adv_dict_list.get("语气副词") \
                          + self.__adv_dict_list.get("情态副词")

        self.__pronoun_list = self.__load_pronoun_words()
        self.__vi_list = self.__load_intransitive_verb()
        self.__auxiliary_dict_list = self.__load_auxiliary_dictionary()
        self.__auxiliary_list = self.__auxiliary_dict_list.get("语气助词") + self.__auxiliary_dict_list.get("结构助词") + self.__auxiliary_dict_list.get("时态助词")

        self.__special_prefix_list = self.__load_special_prefix_words()
        self.__stopwords_list = self.__load_stopwords("之前", "是因为", "已经")

    def release(self):
        self.__labeller.release()
        self.__parser.release()
        self.__postagger.release()
        self.__segmentor.release()

    @classmethod
    def __load_stopwords(cls, *self_define_stopwords):
        """
        get stopwords list
        :param self_define_stopwords: add self define stop word to stopwords list
        :return: stopwords_list
        """
        stopwords_list = [word.strip() for word in open(os.path.join(DICTIONARY_DIR, "stopwords.txt"), "r").readlines()]
        for stopword in self_define_stopwords:
            stopwords_list.append(stopword)
        return stopwords_list

    @classmethod
    def __load_special_prefix_words(cls):
        """
        加载特别开始词
        :return:
        """
        special_prefix_words = []
        with open(os.path.join(DICTIONARY_DIR, "special_prefix.txt"), "r") as sp_file:
            for word in sp_file.readlines():
                special_prefix_words.append(word.strip())
        return special_prefix_words

    @classmethod
    def __load_intransitive_verb(cls):
        """
        加载不及物动词
        :return:
        """
        intransitive_verb = []
        with open(os.path.join(DICTIONARY_DIR, "intransitive_verb.txt"), "r") as vi_file:
            for word in vi_file.readlines():
                intransitive_verb.append(word.strip())
        return intransitive_verb

    @classmethod
    def __load_pronoun_words(cls):
        """
        加载代词
        :return:
        """
        pronoun_words = []
        with open(os.path.join(DICTIONARY_DIR, "pronoun.txt"), "r") as pronoun_file:
            for word in pronoun_file.readlines():
                pronoun_words.append(word.strip())
        return pronoun_words

    @classmethod
    def __load_adverb_dictionary(cls):
        """
        加载副词
        :return:
        """
        dictionary = {}
        with open(os.path.join(DICTIONARY_DIR, "adv.txt"), "r") as adv_file:
            for line in adv_file.readlines():
                index = line.index(":")
                key = line[0: index].strip()
                value = line[index + 1:].strip()
                dictionary.update({key: value.split(" ")})
        return dictionary

    @classmethod
    def __load_auxiliary_dictionary(cls):
        """
        加载助词
        :return:
        """
        dictionary = {}
        with open(os.path.join(DICTIONARY_DIR, "auxiliary.txt"), "r") as adv_file:
            for line in adv_file.readlines():
                index = line.index(":")
                key = line[0: index].strip()
                value = line[index + 1:].strip()
                dictionary.update({key: value.split(" ")})
        return dictionary

    @classmethod
    def __smart_split_sentence(cls, comment):
        """
        拆分句子
        :param comment:
        :return:
        """
        # 替换空格为"，"
        comment = re.sub(re.compile(r"(\s+)", re.S), "，", comment.strip())
        # 句子按分隔[。|！|，|、|？|.|!|,|?]符分出多个子句
        subcomments = re.split(r'[。|！|，|、|？|\.|!|,|\?]', comment)
        return subcomments

    def sentence_segment_add_space(self, comment, stopwords_list={}):
        """
        使用空格间隔分词
        如：
        我们 喜欢 吃 冰激凌
        :param comment: 一条语料
        :param stopwords_list: 停用词列表
        :return:
        """
        self.__segmentor
        segment = self.__segmentor.segment(self.__remove_special_word(comment))
        return segment, " ".join(segment)

    def __word_self_attention(self, parent_pos, parent_word, current_arc_relation, current_arc_pos, current_word):
        """
        判断词性与依存关系组合的有效性

        词注意力机制
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
            if current_arc_relation == Dependency.FOB.value:
                return True
            if current_arc_relation == Dependency.ADV.value:
                if current_arc_pos == Pos.d.value:
                    if current_word in self.__adv_dict_list.get("肯否副词"):
                        return True
                if current_arc_pos == Pos.p.value and current_word in ["由", "用"]:  # 由关晓彤代言
                    return True
                if current_arc_pos == Pos.v.value:
                    return True
            if current_arc_relation == Dependency.ATT.value:
                return True
            if current_arc_relation == Dependency.CMP.value:
                return True
            # if current_arc_pos == Pos.u.value and current_word not in self.__auxiliary_dict_list.get("语气助词") + self.__auxiliary_dict_list.get("时态助词"):
            if current_arc_pos == Pos.u.value and current_word not in self.__auxiliary_list:
                return True
        elif parent_pos == Pos.a.value:
            if current_arc_relation == Dependency.SBV.value and current_word not in self.__pronoun_list:  # e.g.:材料新鲜  它很方便
                return True
            if current_arc_relation == Dependency.ADV.value and (current_word not in self.__adv_dict_list.get("程度副词") + self.__adv_dict_list.get("范围副词")
                                                                 or (current_arc_pos == Pos.p.value and current_word in ["比"])):  # 比别家好
                return True
            if current_arc_relation == Dependency.ATT.value:
                return True
            if current_arc_pos == Pos.u.value and current_word not in self.__auxiliary_dict_list.get("语气助词") + self.__auxiliary_dict_list.get("结构助词"):
                return True
        elif parent_pos in [Pos.n.value, Pos.nd.value, Pos.nh.value, Pos.ni.value, Pos.nl.value, Pos.ns.value, Pos.nt.value, Pos.nz.value]:
            if current_arc_relation == Dependency.ADV.value:
                return True
            if current_arc_relation == Dependency.ATT.value:  # 属性语义修饰名词
                return True
            if current_arc_pos == Pos.u.value and current_word not in self.__auxiliary_dict_list.get("语气助词") + self.__auxiliary_dict_list.get("结构助词"):  # 美丽的
                return True
        elif parent_pos == Pos.p.value:
            if current_arc_relation == Dependency.SBV.value:  # 他给我感觉
                return True
            if current_arc_relation == Dependency.VOB.value:  # 给我感觉
                return True
            if current_arc_relation == Dependency.POB.value:  # 比别家好
                return True
        elif parent_pos == Pos.d.value:
            if current_arc_relation == Dependency.SBV.value:
                return True
            if current_arc_relation == Dependency.VOB.value:  # 没有|d  4|过于|d  5|甜腻
                return True
        elif parent_pos in [Pos.i.value, Pos.r.value, Pos.q.value] or current_arc_relation == Dependency.CMP.value:
            return True
        return False

    def __parse_opinion(self, core_word_index, arcs, words, postags):
        """

        :param core_word_index:
        :param arcs:
        :param words:
        :param postags:
        :return: opinion_word_list
        """
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

        def do_parse_opinion(core_word_idx):
            """
            提取以动词为核心的观点，提取的主要结构主谓结构（SBV）、动宾结构（VOB）、状中结构（ADV）、动补结构（CMP）、介宾结构（POB）
            :return:
            """
            nonlocal has_vob
            nonlocal sbv_word
            nonlocal sbv_att_word_list
            nonlocal available_word_idx_list

            for m, arc in enumerate(arcs):
                # tuple格式：（index, 句法依存关系, 词性, 词）
                current_word_tuple = (m, arc.relation, postags[m], words[m])

                parent_word_index = arc.head - 1
                parent_word_tuple = (parent_word_index, arcs[parent_word_index].relation, postags[parent_word_index], words[parent_word_index])

                if arc.head == core_word_idx + 1 \
                        and (current_word_tuple[2] not in [Pos.wp.value, Pos.o.value, Pos.c.value, Pos.r.value, Pos.e.value] or (current_word_tuple[2] == Pos.r.value and current_word_tuple[3] not in self.__pronoun_list)) \
                        and self.__word_self_attention(parent_word_tuple[2], parent_word_tuple[3], current_word_tuple[1], current_word_tuple[2], current_word_tuple[3]):

                    # 计算词的root词是否等于关键词
                    root_core_index = word_root_index(core_word_index, m)
                    if root_core_index == core_word_index:
                        if arc.relation == Dependency.VOB.value or (arc.relation == Dependency.CMP.value and postags[current_word_tuple[0]] == Pos.a.value):
                            has_vob = True
                            available_word_idx_list.append(m)
                            opinion_word_list.append(current_word_tuple)
                        else:
                            if arc.head - 1 in available_word_idx_list:
                                available_word_idx_list.append(m)
                                # 若是主谓结构先暂存，不加入观点词list
                                if arc.relation == Dependency.SBV.value:
                                    if len(sbv_word) == 0:
                                        sbv_word = current_word_tuple
                                else:
                                    # 计算词的root词是否等于sbv关键词
                                    sbv_index = sbv_word[0] if len(sbv_word) > 0 else -1
                                    root_sbv_index = word_root_index(sbv_index, current_word_tuple[0])
                                    if root_sbv_index == sbv_index:
                                        # 若是主谓结构的其他属性词，暂存在主谓属性词列表
                                        sbv_att_word_list.append(current_word_tuple)
                                    else:
                                        opinion_word_list.append(current_word_tuple)
                    do_parse_opinion(m)

        do_parse_opinion(core_word_index)

        def need_sbv():
            """
            判断是否需要主谓结构
            :return:
            """
            # 三元组判断，只有包含了动宾结构才把主谓结构加入
            if has_vob:
                return True
            # 及物动词可以直接加sbv
            if postags[core_word_index] == Pos.a.value:
                return True
            # 形容词句意可以直接在sbv
            if words[core_word_index] in self.__vi_list:
                return True
            return False

        if need_sbv() and len(sbv_word) > 0:
            opinion_word_list.append(sbv_word)
            opinion_word_list += sbv_att_word_list

        return opinion_word_list

    def extract_opinion(self, comment, distinct_opinion=True, show_core_word=False, show_detail=False):
        """
        抽取观点
        :param comment:
        :param distinct_opinion: 是否去重观点
        :param show_core_word: 是否展示观点核心词
        :param show_detail: 是否展示分词等详细信息
        :return:
        """
        subcomments = self.__smart_split_sentence(comment)
        opinion_list = []
        for subcomment in subcomments:
            words, sentence_with_space = self.sentence_segment_add_space(subcomment)
            opinions = self.__parse_segment(words, show_detail)
            if len(opinions) > 0:
                opinion_list += opinions
        if distinct_opinion:
            opinion_list = self.__distinct_opinion(opinion_list)
        if not show_core_word:
            opinion_list = [opinion[2] for opinion in opinion_list]
        return opinion_list

    @classmethod
    def __distinct_opinion(cls, opinions):
        """
        观点去重
        :param opinions:
        :return:
        """
        index = 2
        distinct_opinion_list = []
        for n in range(1, len(opinions)):
            for m in range(n, 0, -1):
                opi_1 = opinions[m][index]
                opi_2 = opinions[m - 1][index]
                if len(opi_1) > len(opi_2):
                    tmp = opinions[m - 1]
                    opinions[m - 1] = opinions[m]
                    opinions[m] = tmp

        for opinion in opinions:
            opi = opinion[index]
            if len(distinct_opinion_list) == 0:
                distinct_opinion_list.append(opinion)
            else:
                include = False
                for idx in range(0, len(distinct_opinion_list)):
                    try:
                        include |= distinct_opinion_list[idx][index].index(opi) > -1
                    except ValueError:
                        pass
                if not include:
                    distinct_opinion_list.append(opinion)

        return distinct_opinion_list

    def __parse_segment(self, words, show_detail=False):
        postags = self.__postagger.postag(words)

        word_tag_tuple_list = []
        for i in range(len(words)):
            word_tag_tuple_list.append((str(i), words[i], postags[i]))
        arcs = self.__parser.parse(words, postags)

        # arcs 使用依存句法分析的结果
        labels = self.__labeller.label(words, postags, arcs)  # 语义角色标注

        if show_detail:
            logger.info("|".join(words))
            logger.info("  ".join('|'.join(tpl) for tpl in word_tag_tuple_list))
            logger.info("  ".join("%d|%d:%s" % (n, arc.head, arc.relation) for n, arc in enumerate(arcs)))
            for label in labels:
                logger.info(str(label.index) + ":" + ",".join(["%s:(%d,%d)" % (arg.name, arg.range.start, arg.range.end) for arg in label.arguments]))

        # opinions = self.__parse_main_opinion(arcs, words, postags)
        opinions = self.__parse_opinions(arcs, words, postags)
        return opinions

    def __parse_opinions(self, arcs, words, postags):
        """
        给出核心词性，解释所有该词性的短语观点
        :param arcs:
        :param words:
        :param postags:
        :return:
        """
        opinions = []
        for n, arc in enumerate(arcs):
            postag = postags[n]
            word = words[n]
            if postag in [Pos.v.value, Pos.a.value, Pos.i.value] or \
                    (postag == Pos.a.value and word not in self.__adv_list) or \
                    (arc.relation in [Dependency.HED.value, Dependency.COO.value] and postag not in [Pos.v.value, Pos.a.value, Pos.i.value, Pos.m.value, Pos.c.value]):
                opinion_word_list = self.__parse_opinion(n, arcs, words, postags)
                if self.__check_opinion(postag, word, opinion_word_list):
                    opinion_str = self.__opinion_to_str(n, words, opinion_word_list)
                    opinions.append((postag, words[n], opinion_str))

        return opinions

    def __parse_main_opinion(self, arcs, words, postags):
        """

        :param arcs:
        :param words:
        :param postags:
        :return:
        """
        for n, arc in enumerate(arcs):
            if arc.relation == Dependency.HED.value:
                core_index = n
        core_pos = postags[core_index]
        opinion_word_list = self.__parse_opinion(core_index, arcs, words, postags)
        return core_pos, words[core_index], self.__opinion_to_str(core_index, words, opinion_word_list)

    @classmethod
    def __check_opinion(cls, core_word_pos, core_word, opinion_word_list):
        """
        检测opinion有效性
        :param core_word_pos:
        :param core_word:
        :param opinion_word_list:
        :return:
        """
        if len(opinion_word_list) > 0:
            return True
        if len(opinion_word_list) == 0 and core_word_pos not in [Pos.v.value, Pos.d.value]:
            return True
        if len(opinion_word_list) == 0 and core_word_pos == Pos.v.value and len(core_word) > 1:  # 入口即化|v
            return True
        return False

    def __opinion_to_str(self, core_word_index, words, opinion_word_list):
        """
        输出观点字符串
        :param core_word_index:
        :param words:
        :param opinion_word_list:
        :return:
        """
        index_list = [core_word_index]
        if self.__remove_core_word(words[core_word_index]):
            index_list = []

        for opinion_word in opinion_word_list:
            index = opinion_word[0]
            index_list.append(index)
        index_list.sort()

        opinion = ""
        for index in index_list:
            opinion += words[index]

        return self.__remove_special_word(opinion)

    @classmethod
    def __remove_core_word(cls, word):
        if word == "是":
            return True
        return False

    def __remove_special_word(self, opinion):
        new_opinion = opinion
        for sp_word in self.__special_prefix_list:
            if opinion.rfind(sp_word) == 0:
                new_opinion = opinion[len(sp_word):]
                return self.__remove_special_word(new_opinion)
        return new_opinion


def handle_comment():
    import json
    comment_list = []
    with open("../data/comment", "r", encoding="utf-8") as comments:
        for comment in comments:
            comment_list.append(comment.strip())
    comments = json.dumps({"comment": comment_list}, ensure_ascii=False)
    print(str(comments))


def main():
    opinion_extractor = OpinionExtractor()
    # with open("../data/comment4", "r", encoding="utf-8") as comments:
    #     for comment in comments:
    #         opinions = opinion_extractor.extract_opinion(comment)
    #         print(comment.strip(), ":", opinions)

    # print(opinion_extractor.extract_opinion("我觉得不是这样，清洁干净只是暂时的干净", show_core_word=True, show_detail=True))
    # print(opinion_extractor.extract_opinion("动物奶油制作，感觉比较健康，吃起来也没有过于甜腻", show_detail=True))
    # print(opinion_extractor.extract_opinion("关晓彤贯穿始终", show_detail=True))
    # print(opinion_extractor.extract_opinion("不会很腻", show_detail=True))
    # print(opinion_extractor.extract_opinion("杯子颜色清爽，干净，代言人青春美少女", show_detail=True))

    # print(opinion_extractor.extract_opinion("因为它的代言人挺好看的，它的外观也挺好看的，色调属于蓝绿色吧，也挺不错的，挺舒适的", show_detail=True))
    # print(opinion_extractor.extract_opinion("包装是很好看的颜色 色彩很鲜明", show_detail=True))
    # print(opinion_extractor.extract_opinion("包装和广告词语", show_detail=True))
    # print(opinion_extractor.extract_opinion("极不专业的动作，舞蹈和鼓都是", show_core_word=True, show_detail=True))
    # print(opinion_extractor.extract_opinion("比较青春活力", show_core_word=True, show_detail=True))
    # print(opinion_extractor.extract_opinion("是应该比较出名的小姑娘", show_core_word=True, show_detail=True))
    logger.info(opinion_extractor.extract_opinion("不腻", distinct_opinion=True, show_core_word=True, show_detail=True))
    opinion_extractor.release()


if __name__ == '__main__':
    handle_comment()