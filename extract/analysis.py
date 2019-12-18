import random
from pyltp import Segmentor, Postagger, Parser, SementicRoleLabeller

from gensim.models import KeyedVectors
from pyhanlp import *
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer

from sentiment.pos import Pos
from sentiment.dependency import Dependency
import numpy as np
import re

# LTP_MODEL_DIR = "E:/02program/python/nlp/data/ltp_data_v3.4.0"
LTP_MODEL_DIR = "/Users/zhuzhibin/Program/python/qd/nlp/nlp-platform/opinion-data/ltp-model"

DICTIONARY_DIR = "/Users/zhuzhibin/Program/python/qd/nlp/nlp-platform/opinion-data/opn-model"

segmentor = Segmentor()
postagger = Postagger()
parser = Parser()  # 初始化实例
labeller = SementicRoleLabeller()  # 初始化实例

segmentor.load_with_lexicon(os.path.join(LTP_MODEL_DIR, "cws.model"), os.path.join(DICTIONARY_DIR, "custom_lexicon.model"))
postagger.load_with_lexicon(os.path.join(LTP_MODEL_DIR, "pos.model"), os.path.join(DICTIONARY_DIR, "custom_lexicon.model"))
parser.load(os.path.join(LTP_MODEL_DIR, "parser.model"))  # 加载模型
labeller.load(os.path.join(LTP_MODEL_DIR, "pisrl.model"))  # 加载模型

train = []
test = []

negative_feature_words = ["不", "没有", "没", "无", "暂时"]
auxiliary_feature_words = ["吗", "呢", "吧", "啊", "呀", "哇", "哦", "哪", "啦", "噢", "叭", "咦", "哈"]
verb_ignore_feature_words = ["会", "是", "有", "要"]

special_prefix_model = "/Users/zhuzhibin/Program/python/qd/nlp/nlp-platform/opinion-data/opn-model/special_prefix.model"


def load_special_prefix_words(special_prefix_model):
    """
    加载特别开始词
    :return:
    """
    special_prefix_words = []
    with open(special_prefix_model, "r", encoding="utf-8") as sp_file:
        for word in sp_file.readlines():
            special_prefix_words.append(word.strip())
    return special_prefix_words


special_prefix_list = load_special_prefix_words(special_prefix_model)


def remove_special_word(doc_segment):
    """
    移除特殊词
    :param self:
    :param doc:
    :return:
    """
    first_term = doc_segment[0]
    if first_term in special_prefix_list:
        doc_segment.pop(0)
        return remove_special_word(doc_segment)
    return doc_segment


def doc_split(doc):
    doc = re.sub(re.compile(r"(\s+)", re.S), "，", doc.strip())
    pattern = "[" + "|".join(auxiliary_feature_words) + "]"
    doc = re.sub(re.compile(pattern, re.S), "", doc.strip())
    # doc = re.sub(re.compile(r"[吗|呢|吧|啊|呀|哇|哦|哪|啦|噢|叭|咦|哈|的]", re.S), "", doc.strip())
    sub_doc_list = re.split(r'[。|！|，|、|？|\.|!|,|\?]', doc)
    sub_doc_segment_list = []
    for sub_doc in sub_doc_list:
        doc_segment = remove_special_word([term for term in segmentor.segment(sub_doc)])
        sub_doc_segment_list.append("".join(doc_segment))
    return sub_doc_segment_list


def smart_doc_split(doc):
    """智能分句"""
    doc = re.sub(re.compile(r"(\s+)", re.S), "，", doc.strip())
    pattern = "[" + "|".join(auxiliary_feature_words) + "]"
    doc = re.sub(re.compile(pattern, re.S), "", doc.strip())
    # doc = re.sub(re.compile(r"[吗|呢|吧|啊|呀|哇|哦|哪|啦|噢|叭|咦|哈|的]", re.S), "", doc.strip())
    sub_doc_list = []

    def format_doc(doc):
        _format_doc = re.sub(re.compile(r"[。|！|，|、|？|\.|!|,|\?]", re.S), "", doc.strip())
        return _format_doc

    def term_distance(core_id, idx):
        return np.math.fabs(core_id - idx)

    def term_root_idx_search(core_idx, idx, before_relation, hed_pos, segment, dependency):
        """
        查找词的root index
        :return:
        """
        node = dependency[idx]
        term = segment[idx]
        index = idx if node.relation == Dependency.HED.value \
                       or (node.relation == Dependency.COO.value and (before_relation != Dependency.LAD.value and term not in ["只是", "是"])) else node.head - 1
        if index == core_idx or index == idx:
            return index
        else:
            return term_root_idx_search(core_idx, index, node.relation, hed_pos, segment, dependency)

    def do_smart_split(document):
        term_list = []
        core_idx = None
        segment = segmentor.segment(document)
        postag = postagger.postag(segment)
        dependency = parser.parse(segment, postag)

        hed_pos = None
        for n, node in enumerate(dependency):
            if node.relation == Dependency.HED.value:
                core_idx = n
                hed_pos = postag[core_idx]
                break

        before_relation = None
        new_document_term_list = []
        # 处理COO的附和问题
        core_idx_list = [core_idx]
        for idx, node in enumerate(dependency):
            # term是否已被使用
            used = False
            pos = postag[idx]
            term = segment[idx]
            relation = node.relation
            if (relation == Dependency.COO.value and (before_relation == Dependency.LAD.value or term in ["只是", "是"])) or relation != Dependency.COO.value:
                same_root = False
                for __core_idx in core_idx_list:
                    root_idx = term_root_idx_search(__core_idx, idx, before_relation, hed_pos, segment, dependency)
                    if root_idx == __core_idx:
                        same_root = True
                        break
                if same_root:
                    if before_relation == Dependency.WP.value and relation == Dependency.WP.value:
                        continue
                    before_relation = relation
                    term_list.append(segment[idx])
                    used = True
                # 增加"LAD"后的"COO"词为核心词，即右附和词
                if relation == Dependency.COO.value:
                    core_idx_list.append(idx)
            if not used:
                new_document_term_list.append(term)

        new_document = "".join(new_document_term_list)

        # lad_doc = False
        # for idx, term in enumerate(term_list):
        #     if dependency[idx].relation == Dependency.LAD.value:
        #         left_sub_doc = "".join(term_list[0:idx])
        #         right_sub_doc = "".join(term_list[idx + 1:len(term_list)])
        #         sub_doc_list.append(format_doc(left_sub_doc))
        #         sub_doc_list.append(format_doc(right_sub_doc))
        #         lad_doc = True
        #         break
        # if not lad_doc:
        sub_doc = "".join(term_list)
        sub_doc_list.append(format_doc(sub_doc))

        if len(new_document) > 0:
            return do_smart_split(new_document)

    do_smart_split(doc)
    return sub_doc_list


class Term(object):
    def __init__(self, pos, term, relation, relative_idx):
        self.pos = pos
        self.term = term
        self.relation = relation
        if term is not None:
            self.relation = term + "#" + relation
        self.relative_idx = relative_idx


class Template(object):
    def __init__(self):
        self.src_pattern_term_list = []
        self.target_pattern_term_list = []

    def add_src_term(self, term):
        self.src_pattern_term_list.append(term)

    def add_target_term(self, term):
        self.target_pattern_term_list.append(term)

    def template_src_pattern(self):
        src_term = []
        for term in self.src_pattern_term_list:
            src_term.append(term.pos)
            src_term.append(term.relation)
        if len(src_term) == 0:
            return ""
        return "-".join(src_term)

    def template_target_pattern(self):
        target_term = []
        for term in self.target_pattern_term_list:
            target_term.append(term.pos)
            target_term.append(term.relation)
        if len(target_term) == 0:
            return ""
        return "-".join(target_term)


def load_corpus():
    with open("../data/tag-data/why-tag.txt", "r", encoding="utf-8") as tag_data:
        lines = [line for line in tag_data.readlines()]
        size = len(lines)
        train_size = int(size * 0.8)
        test_size = size - train_size
        train_idxs = [idx for idx in random.sample(range(0, size), train_size)]
        for n, line in enumerate(lines):
            if n in train_idxs:
                train.append(line)
            else:
                test.append(line)

        print("train size: ", len(train), "test_size: ", len(test))


def segment_benchmark():
    match_ltp = 0
    match_jieba = 0
    match_hanlp = 0

    total = len(train)
    for train_line in train:
        doc = train_line.split(":")[0].strip()
        opinion = train_line.split(":")[1].split(",")[0].strip()

        # ltp分词
        ltp_doc_segment = [word for word in segmentor.segment(doc)]
        ltp_opinion_segment = [word for word in segmentor.segment(opinion)]
        if check(ltp_opinion_segment, ltp_doc_segment):
            match_ltp += 1

        # hanlp分词
        hanlp_doc_segment = [term.word for term in HanLP.segment(doc)]
        hanlp_opinion_segment = [term.word for term in HanLP.segment(opinion)]
        if check(hanlp_opinion_segment, hanlp_doc_segment):
            match_hanlp += 1

        # jieba分词
        jb_doc_segment = [term for term in jieba.cut(doc)]
        jb_opinion_segment = [term for term in jieba.cut(opinion)]
        if check(jb_opinion_segment, jb_doc_segment):
            match_jieba += 1

    print("ltp marks: ", str(match_ltp / total * 100), " hanlp marks: ", str(match_hanlp / total * 100), " jieba marks: ", str(match_jieba / total * 100))


def train_model():
    """
    {
        p1:{
                count:3,
                p1-1:{object:template,count:1}
                p1-2:{object:template,count:2}
            }
    }
    :return:
    """
    template_src_dict = {}
    for train_line in train:
        doc = train_line.split(":")[0].strip()
        opinion = train_line.split(":")[1].split(",")[0].strip()

        # ltp分词
        doc_segment = [word for word in segmentor.segment(doc)]
        opinion_segment = [word for word in segmentor.segment(opinion)]

        if check(opinion_segment, doc_segment):
            # 左匹配
            left_start_idx = len(doc_segment) - 1
            left_end_idx = 0
            for opinion_term in opinion_segment:
                for idx, doc_term in enumerate(doc_segment):
                    if opinion_term == doc_term:
                        if idx < left_start_idx:
                            left_start_idx = idx
                        if idx > left_end_idx:
                            left_end_idx = idx

            left_distance = left_end_idx - left_start_idx

            # 右匹配
            right_start_idx = len(doc_segment) - 1
            right_end_idx = 0
            for opinion_term in reversed(opinion_segment):
                for idx, doc_term in enumerate(doc_segment):
                    if opinion_term == doc_term:
                        if idx < right_start_idx:
                            right_start_idx = idx
                        if idx > right_end_idx:
                            right_end_idx = idx

            right_distance = right_end_idx - right_start_idx

            if left_distance > right_distance:
                start_idx = right_start_idx
                end_idx = right_end_idx
            else:
                start_idx = left_start_idx
                end_idx = left_end_idx

            n_start_idx, n_end_idx = dependency_n_grim(1, start_idx, end_idx, len(doc_segment))

            postag = postagger.postag(doc_segment)
            dependency = parser.parse(doc_segment, postag)

            postag = [pos for pos in postag]
            dependency = [node for node in dependency]
            dependency = dependency[n_start_idx:n_end_idx + 1]

            # 写入模板
            template = Template()
            for offset, node in enumerate(dependency):
                idx = offset + n_start_idx
                term = doc_segment[idx]
                pos = postag[idx]
                if pos.startswith(Pos.n.value):
                    pos = Pos.n.value
                relation = node.relation

                if term in negative_feature_words:
                    term_object = Term(pos, term, relation, offset)
                else:
                    term_object = Term(pos, None, relation, offset)

                template.add_src_term(term_object)

                if term in opinion_segment:
                    template.add_target_term(term_object)

            if not template_src_dict.__contains__(template.template_src_pattern()):
                template_src_dict[template.template_src_pattern()] = {"count": 0}
            template_src_dict[template.template_src_pattern()]["count"] += 1

            if not template_src_dict[template.template_src_pattern()].__contains__(template.template_target_pattern()):
                template_src_dict[template.template_src_pattern()][template.template_target_pattern()] = {"count": 0, "template": None}
            template_src_dict[template.template_src_pattern()][template.template_target_pattern()]["template"] = template
            template_src_dict[template.template_src_pattern()][template.template_target_pattern()]["count"] += 1

    template_src_item_list = [item for item in template_src_dict.items()]
    template_src_item_list.sort(key=lambda item: item[1]["count"], reverse=True)

    best_template_src_item_list = []
    for item in template_src_item_list:
        # 大于2认为是好的
        if item[1]["count"] >= 1:
            best_template_src_item_list.append(item)

    best_template_list = []
    for item in best_template_src_item_list:
        global_best = None
        src_pattern = item[0]
        target_dict = item[1]
        src_count = target_dict["count"]
        for target_pattern, target_pattern_dict in target_dict.items():
            if target_pattern != "count" and len(target_pattern) > 0:
                target_template = target_pattern_dict["template"]
                target_count = target_pattern_dict["count"]
                best = target_count / src_count * 100
                if global_best is None:
                    global_best = best, target_template
                else:
                    if best > global_best[0]:
                        global_best = best, target_template
        best_template_list.append(global_best[1])
    best_template_list.sort(key=lambda template: len(template.src_pattern_term_list), reverse=True)

    return best_template_list


def show_model(best_template_list):
    for template in best_template_list:
        print("-".join([term.pos + "-" + term.relation for term in template.src_pattern_term_list]), ":", "-".join([str(term.relation) for term in template.target_pattern_term_list]))


def process_corpus():
    global_distance = 0
    seq_dict = {}
    for train_line in train:
        doc = train_line.split(":")[0].strip()
        opinion = train_line.split(":")[1].split(",")[0].strip()

        # ltp分词
        doc_segment = [word for word in segmentor.segment(doc)]
        opinion_segment = [word for word in segmentor.segment(opinion)]
        # hanlp分词
        # doc_segment = [term.word for term in HanLP.segment(doc)]
        # opinion_segment = [term.word for term in HanLP.segment(opinion)]
        # jieba分词
        # doc_segment = [term for term in jieba.cut(doc)]
        # opinion_segment = [term for term in jieba.cut(opinion)]

        # print("doc_segment: ", doc_segment, "  opinion_segment: ", opinion_segment)

        if check(opinion_segment, doc_segment):
            # 左匹配
            left_start_idx = len(doc_segment)
            left_end_idx = 0
            for opinion_term in opinion_segment:
                for idx, doc_term in enumerate(doc_segment):
                    if opinion_term == doc_term:
                        if idx < left_start_idx:
                            left_start_idx = idx
                        if idx > left_end_idx:
                            left_end_idx = idx

            left_distance = left_end_idx - left_start_idx

            # 右匹配
            right_start_idx = len(doc_segment)
            right_end_idx = 0
            for opinion_term in reversed(opinion_segment):
                for idx, doc_term in enumerate(doc_segment):
                    if opinion_term == doc_term:
                        if idx < right_start_idx:
                            right_start_idx = idx
                        if idx > right_end_idx:
                            right_end_idx = idx

            right_distance = right_end_idx - right_start_idx

            start_idx = 0
            end_idx = 0
            distance = 0
            if left_distance > right_distance:
                start_idx = right_start_idx
                end_idx = right_end_idx
                distance = right_distance
            else:
                start_idx = left_start_idx
                end_idx = left_end_idx
                distance = left_distance

            global_distance = np.max([distance, global_distance])

            start_idx, end_idx = dependency_n_grim(1, start_idx, end_idx, len(doc_segment))

            postag = postagger.postag(doc_segment)
            dependency = parser.parse(doc_segment, postag)
            postag = [pos for pos in postag]
            dependency = [node for node in dependency]
            dependency = dependency[start_idx:end_idx + 1]

            doc_seq_list = []
            opinion_seq_list = []
            for idx, node in enumerate(dependency):
                new_idx = idx + start_idx
                term = doc_segment[new_idx]
                pos = postag[new_idx]
                if pos.startswith("n"):
                    pos = Pos.n.value
                pos_posistion = pos + "[" + str(idx) + "]"
                relation = node.relation
                relation_posistion = relation + "[" + str(idx) + "]"

                if term in negative_feature_words:
                    doc_seq_list.append(term)
                # doc_seq_list.append(pos_posistion)
                doc_seq_list.append(relation_posistion)

                # elif relation not in [Dependency.ADV.value, Dependency.ATT.value]:
                #     # seq_list.append(pos)
                #     doc_seq_list.append(relation)

                if term in opinion_segment:

                    if term in negative_feature_words:
                        opinion_seq_list.append(term)
                    # opinion_seq_list.append(pos_posistion)
                    opinion_seq_list.append(relation_posistion)

                # seq_list.append(pos)
                # if relation == Dependency.ADV.value and word == "不":
                #     seq_list.append("不")
                # seq_list.append(relation)

            doc_seq = "-".join(doc_seq_list)
            opinion_seq = "-".join(opinion_seq_list)
            inner_dict = {}
            if seq_dict.__contains__(doc_seq):
                inner_dict = seq_dict[doc_seq]
                inner_dict["size"] += 1
                if inner_dict.__contains__(opinion_seq):
                    inner_dict[opinion_seq] += 1
                else:
                    inner_dict[opinion_seq] = 1
            else:
                inner_dict["size"] = 1
                inner_dict[opinion_seq] = 1
                seq_dict[doc_seq] = inner_dict

    print("distance: ", global_distance)

    seq_item_list = [item for item in seq_dict.items()]
    seq_item_list.sort(key=lambda seq: seq[1]["size"], reverse=True)

    best_template_list = []
    for item in seq_item_list:
        if item[1]["size"] >= 2:
            best_template_list.append(item)

    best_template_list.sort(key=lambda seq: len(seq[0].split("-")), reverse=True)

    for item in best_template_list:
        doc_seq = item[0]
        inner_dict = item[1]
        size = inner_dict["size"]

        global_best = None
        for inner_item in inner_dict.items():
            inner_item_key = inner_item[0]
            inner_item_value = inner_item[1]
            if inner_item_key != "size" and len(inner_item_key) > 0:
                best = inner_item_value / size * 100
                if global_best is None:
                    global_best = inner_item
                else:
                    if best > global_best[1] / size * 100:
                        global_best = inner_item
        print(doc_seq, "(", size, "):", global_best)
        # print(item)

    model = {}
    for item in best_template_list:
        doc_seq = item[0]
        inner_dict = item[1]
        size = inner_dict["size"]

        global_best = None
        for inner_item in inner_dict.items():
            inner_item_key = inner_item[0]
            inner_item_value = inner_item[1]
            if inner_item_key != "size" and len(inner_item_key) > 0:
                best = inner_item_value / size * 100
                if global_best is None:
                    global_best = inner_item
                else:
                    if best > global_best[1] / size * 100:
                        global_best = inner_item


def check(opinion_segment, doc_segment):
    """
    检测分词是否一致
    :param opinion_segment:
    :param doc_segment:
    :return:
    """
    match_size = 0
    opinion_size = len(opinion_segment)
    for opinion_term in opinion_segment:
        for doc_term in doc_segment:
            if opinion_term == doc_term:
                match_size += 1
                break
    if match_size == opinion_size:
        return True
    return False


def dependency_n_grim(n, start, end, length):
    new_start = start
    if start - n >= 0:
        new_start = start - n

    new_end = end
    if end + n < length:
        new_end = end + n

    return new_start, new_end


def extract_opinion(comment, model):
    segment = segmentor.segment(comment)
    postag = postagger.postag(segment)
    dependency = parser.parse(segment, postag)

    postag = [pos for pos in postag]
    dependency = [node for node in dependency]

    length = len(dependency)
    opinion = None
    for template in model:
        compare_pattern = template.template_src_pattern()
        pattern_length = len(template.src_pattern_term_list)

        for idx, node in enumerate(dependency):
            start_idx = idx
            end_idx = idx + pattern_length - 1
            if end_idx < length:
                pattern_term_list = []
                for n in range(start_idx, end_idx + 1):
                    pattern_term_list.append(postag[n])
                    pattern_term_list.append(segment[n] + "#" + dependency[n].relation if segment[n] in negative_feature_words else dependency[n].relation)
                pattern = "-".join(pattern_term_list)
                if pattern == compare_pattern:
                    target_term_list = template.target_pattern_term_list
                    for term in target_term_list:
                        if opinion is None:
                            opinion = segment[term.relative_idx + idx]
                        else:
                            opinion += segment[term.relative_idx + idx]
                    return template, opinion
            else:
                break
    return template, None


def compute_f1_score(corpus, model, show_detail=False):
    template_predict_dict = {}
    for doc in corpus:
        comment = doc.split(":")[0].strip()
        compare_opinion = doc.split(":")[1].split(",")[0].strip()
        template, opinion = extract_opinion(comment, model)

        pattern = template.template_src_pattern()

        if not template_predict_dict.__contains__(pattern):
            template_predict_dict[pattern] = [0, 0]

        if opinion == compare_opinion:
            # 匹配
            template_predict_dict[pattern][0] += 1
        else:
            # 不匹配
            template_predict_dict[pattern][1] += 1
        if show_detail:
            print("comment:", comment, " compare opinion:", compare_opinion, " opinion:", opinion)

    total_tp_size = sum([item[1][0] for item in template_predict_dict.items()])
    total_fp_size = sum([item[1][1] for item in template_predict_dict.items()])
    precision = total_tp_size / (total_tp_size + total_fp_size)

    print("precision:", precision)

    predict_pattern_size = len(template_predict_dict)
    recall = sum([item[1][0] / (item[1][0] + item[1][1]) for item in template_predict_dict.items()]) / predict_pattern_size

    print("recall:", recall)

    f1 = 2 * precision * recall / (precision + recall) * 100
    return f1


def extract_why_opinion(comment):
    opinion_list = []
    sub_doc_list = smart_doc_split(comment)
    for sub_doc in sub_doc_list:
        segment = segmentor.segment(sub_doc)
        postag = postagger.postag(segment)
        dependency = parser.parse(segment, postag)
        opinion = extract_why_rule(segment, postag, dependency)
        if len(opinion) > 0:
            opinion_list.append(opinion)
    return opinion_list


def extract_which_opinion(comment):
    opinion_list = []
    sub_doc_list = smart_doc_split(comment)
    for sub_doc in sub_doc_list:
        segment = segmentor.segment(sub_doc)
        postag = postagger.postag(segment)
        dependency = parser.parse(segment, postag)
        opinion = extract_which_rule(segment, postag, dependency)
        if len(opinion) > 0:
            opinion_list.append("".join(opinion))
    return opinion_list


def extract_why_rule(segment, postag, dependency):
    def term_root_index(core_idx, idx, dependency, parent=True):
        """
        查找词的root index
        :return:
        """
        if parent:
            dependency_node = dependency[idx]
            index = idx if dependency_node.relation in [Dependency.HED.value, Dependency.COO.value] else dependency_node.head - 1
            if index == core_idx or index == idx:
                return index
            else:
                return term_root_index(core_idx, index, dependency, parent)
        else:
            find_root = False
            index = idx
            for node_idx, node in enumerate(dependency):
                if node.head - 1 == index:
                    find_root = True
                    index = node_idx
                    break

            if not find_root:
                return idx
            if index == core_idx:
                return core_idx
            else:
                return term_root_index(core_idx, index, dependency, parent)

    match = False
    term_list = []
    # 专有名词模板
    length = len(segment)
    for idx in range(0, length):
        term = segment[idx]
        pos = postag[idx]
        # i 类型的词要大于2个字，否则当作形容词
        if pos in [Pos.nh.value, Pos.j.value, Pos.nz.value, Pos.ni.value, Pos.z.value] or (pos == Pos.i.value and len(term) > 2):
            term_list = [term]
            match = True
            break

    if not match:
        # 形容词模板
        for idx in range(0, length):
            term = segment[idx]
            pos = postag[idx]
            dependency_node = dependency[idx]
            # i 类型的词要小于等于2个字
            if (pos == Pos.a.value or (pos == Pos.i.value and len(term) <= 2)) and dependency_node.relation != Dependency.ADV.value:
                core_adj_idx = idx
                core_term_idx = 0

                # 检测下个词是否是形容词
                new_core_adj_idx = core_adj_idx
                for next_idx in range(min(length, core_adj_idx + 1), length):
                    if postag[next_idx] == Pos.a.value and new_core_adj_idx + 1 == next_idx:
                        new_core_adj_idx = next_idx
                    else:
                        break
                core_adj_idx = new_core_adj_idx
                core_term = segment[core_adj_idx]
                term_list = [core_term]

                # n 是 a结构，转移"是"的索引为core索引
                if dependency_node.relation == Dependency.VOB.value:
                    if segment[dependency_node.head - 1] == "是":
                        core_adj_idx = dependency_node.head - 1

                for n_idx in range(core_adj_idx - 1, -1, -1):
                    term = segment[n_idx]
                    pos = postag[n_idx]
                    dependency_node = dependency[n_idx]
                    if term in negative_feature_words:
                        root_id = term_root_index(core_adj_idx, n_idx, dependency)
                        if root_id != core_adj_idx:
                            root_id = term_root_index(core_adj_idx, n_idx, dependency, False)
                        if root_id == core_adj_idx:
                            term_list.insert(0, term)
                            core_term_idx += 1
                    if pos == Pos.n.value and dependency_node.relation != Dependency.POB.value:
                        root_id = term_root_index(core_adj_idx, n_idx, dependency)
                        if root_id != core_adj_idx:
                            root_id = term_root_index(core_adj_idx, n_idx, dependency, False)
                        if root_id == core_adj_idx:
                            term_list.insert(0, term)
                            core_term_idx += 1
                            match = True
                            break
                for n_idx in range(core_adj_idx + 1, length):
                    term = segment[n_idx]
                    pos = postag[n_idx]
                    if term in negative_feature_words:
                        root_id = term_root_index(core_adj_idx, n_idx, dependency)
                        if root_id != core_adj_idx:
                            root_id = term_root_index(core_adj_idx, n_idx, dependency, False)
                        if root_id == core_adj_idx:
                            term_list.insert(0, term)
                            core_term_idx += 1
                    if pos == Pos.n.value and dependency_node.relation != Dependency.POB.value:
                        root_id = term_root_index(core_adj_idx, n_idx, dependency)
                        if root_id != core_adj_idx:
                            root_id = term_root_index(core_adj_idx, n_idx, dependency, False)
                        if root_id == core_adj_idx:
                            term_list.insert(0, term)
                            core_term_idx += 1
                            match = True
                            break

                # 形容词的以上规则没找到则找否定词
                if not match:
                    term_list = [core_term]
                    for negative_idx in range(0, length):
                        term = segment[negative_idx]
                        if term in negative_feature_words:
                            root_id = term_root_index(core_adj_idx, negative_idx, dependency)
                            if root_id != core_adj_idx:
                                root_id = term_root_index(core_adj_idx, negative_idx, dependency, False)
                            if root_id == core_adj_idx:
                                term_list.insert(0, term)
                                core_term_idx += 1
                                match = True

                if not match:
                    term_list = [core_term]
                break

    if not match:
        pass
    return term_list


def term_root_idx_full_search(core_idx, idx, dependency):
    """
    双向查找词的root索引
    :param core_idx:
    :param idx:
    :param dependency:
    :return: root id
    """
    index = term_root_idx_search(core_idx, idx, dependency)
    if index != core_idx:
        index = term_root_idx_search(core_idx, idx, dependency, False)
    return index


def term_root_idx_search(core_idx, idx, dependency, parent=True):
    def do_term_root_idx_search(_core_idx, _idx, _dependency):
        _dependency_node = _dependency[_idx]
        _index = _idx if _dependency_node.relation in [Dependency.HED.value, Dependency.COO.value] else _dependency_node.head - 1
        if _index == _core_idx or _index == _idx:
            return _index
        else:
            return do_term_root_idx_search(_core_idx, _index, _dependency)

    core_index = core_idx
    index = idx
    if not parent:
        core_index = idx
        index = core_idx

    _root_id = do_term_root_idx_search(core_index, index, dependency)

    if core_index == _root_id:
        return core_idx
    else:
        return _root_id


def extract_which_rule(segment, postag, dependency):
    """
        以名词为核心词
        主要模式：ATT+n，v+n，a+n
        :param docs:
        :return:
        """
    match = False
    negative_term_count = 0
    length = len(segment)
    term_list = []

    for idx in range(0, length):
        term = segment[idx]
        pos = postag[idx]

        if pos in [Pos.nh.value, Pos.nt.value, Pos.j.value, Pos.nz.value, Pos.ni.value, Pos.z.value]:
            term_list = [term]
            match = True
            break

    if not match:
        # 名词+形容词模板
        for idx in range(0, length):
            term = segment[idx]
            pos = postag[idx]
            if pos == Pos.b.value:
                pos = Pos.n.value
            dependency_node = dependency[idx]
            if not match and pos == Pos.n.value and dependency_node.relation == Dependency.SBV.value:
                core_n_idx = idx
                core_term = term

                negative_term_count = 0
                core_term_idx = 0
                term_list = [core_term]

                for adj_idx in range(0, length):
                    term = segment[adj_idx]
                    pos = postag[adj_idx]
                    if term in negative_feature_words:
                        root_id = term_root_idx_full_search(core_n_idx, adj_idx, dependency)
                        if root_id == core_n_idx:
                            term_list.insert(core_term_idx, term)
                            core_term_idx += 1
                            negative_term_count += 1
                    if pos == Pos.a.value:
                        root_id = term_root_idx_full_search(core_n_idx, adj_idx, dependency)
                        if root_id == core_n_idx:
                            if adj_idx < core_n_idx:
                                term_list.insert(core_term_idx, term)
                                core_term_idx += 1
                            else:
                                term_list.append(term)
                            match = True

    if not match:
        # 名词（ATT）+名词模板
        for idx in range(0, length):
            term = segment[idx]
            pos = postag[idx]
            if pos == Pos.b.value:
                pos = Pos.n.value

            if not match and pos == Pos.n.value and dependency_node.relation != Dependency.ATT.value:
                core_n_idx = idx
                core_term = term

                negative_term_count = 0
                core_term_idx = 0
                term_list = [core_term]

                for n_idx in range(0, length):
                    term = segment[n_idx]
                    pos = postag[n_idx]
                    if pos == Pos.b.value:
                        pos = Pos.n.value
                    dependency_node = dependency[n_idx]
                    if pos == Pos.n.value and dependency_node.relation == Dependency.ATT.value:
                        root_id = term_root_idx_full_search(core_n_idx, n_idx, dependency)
                        if root_id == core_n_idx:
                            if n_idx < core_n_idx:
                                term_list.insert(core_term_idx, term)
                                core_term_idx += 1
                            else:
                                term_list.append(term)
                            match = True
    if not match:
        # 名词（VOB/FOB）+动词模板
        for idx in range(0, length):
            term = segment[idx]
            pos = postag[idx]
            if pos == Pos.b.value:
                pos = Pos.n.value
            dependency_node = dependency[idx]
            if not match and pos == Pos.n.value and dependency_node.relation in [Dependency.VOB.value, Dependency.FOB.value]:
                core_n_idx = idx
                core_term = term

                negative_term_count = 0
                core_term_idx = 0
                term_list = [core_term]

                for v_idx in range(0, length):
                    term = segment[v_idx]
                    pos = postag[v_idx]
                    v_dependency_node = dependency[idx]
                    if term in negative_feature_words:
                        root_id = term_root_idx_full_search(core_n_idx, v_idx, dependency)
                        if root_id == core_n_idx:
                            term_list.insert(core_term_idx, term)
                            core_term_idx += 1
                            negative_term_count += 1
                    if pos == Pos.v.value and v_dependency_node == Dependency.HED.value:
                        root_id = term_root_idx_full_search(core_n_idx, v_idx, dependency)
                        if root_id == core_n_idx:
                            if v_idx < core_n_idx:
                                term_list.insert(core_term_idx, term)
                                core_term_idx += 1
                            else:
                                term_list.append(term)
                            match = True

        if not match:
            # 名词（SBV）+动词模板
            for idx in range(0, length):
                term = segment[idx]
                pos = postag[idx]
                if pos == Pos.b.value:
                    pos = Pos.n.value
                dependency_node = dependency[idx]
                if not match and pos == Pos.n.value and dependency_node.relation == Dependency.SBV.value:
                    core_n_idx = idx
                    core_term = term

                    negative_term_count = 0
                    core_term_idx = 0
                    term_list = [core_term]

                    for v_idx in range(0, length):
                        term = segment[v_idx]
                        pos = postag[v_idx]
                        v_dependency_node = dependency[idx]
                        if term in negative_feature_words:
                            root_id = term_root_idx_full_search(core_n_idx, v_idx, dependency)
                            if root_id == core_n_idx:
                                term_list.insert(core_term_idx, term)
                                core_term_idx += 1
                                negative_term_count += 1
                        if pos == Pos.v.value and v_dependency_node == Dependency.HED.value:
                            root_id = term_root_idx_full_search(core_n_idx, v_idx, dependency)
                            if root_id == core_n_idx:
                                if v_idx < core_n_idx:
                                    term_list.insert(core_term_idx, term)
                                    core_term_idx += 1
                                else:
                                    term_list.append(term)
                                match = True

    if not match:
        for idx in range(0, length):
            term = segment[idx]
            pos = postag[idx]
            if pos == Pos.b.value:
                pos = Pos.n.value
            dependency_node = dependency[idx]
            if not match and pos == Pos.n.value and dependency_node.relation == Dependency.HED.value:
                term_list = [term]
                break;

    result_term_list = []
    if negative_term_count == 2:
        for term in term_list:
            if term not in negative_feature_words:
                result_term_list.append((term, pos))
    else:
        result_term_list = term_list
    return result_term_list


def extract_why_rule(segment, postag, dependency):
    match = False
    term_list = []
    # 专有名词模板
    length = len(postag)
    for n, pos in enumerate(postag):
        term = segment[n]
        # i 类型的词要大于2个字，否则当作形容词
        if pos in [Pos.nh.value, Pos.j.value, Pos.nz.value, Pos.ni.value] or (pos == Pos.i.value and len(term) > 2):
            term_list = [term]
            match = True
            break

    if not match:
        # 形容词模板
        for idx, pos in enumerate(postag):
            term = segment[idx]
            # i 类型的词要小于等于2个字
            if pos == Pos.a.value or (pos == Pos.i.value and len(term) <= 2):
                core_adj_idx = idx
                core_term_idx = 0
                # 检测下个词是否是形容词
                new_core_adj_idx = core_adj_idx
                for next_idx in range(min(length, core_adj_idx + 1), length):
                    if postag[next_idx] == Pos.a.value and new_core_adj_idx + 1 == next_idx:
                        term = segment[next_idx]
                        new_core_adj_idx = next_idx
                    else:
                        break
                core_adj_idx = new_core_adj_idx
                core_term = segment[core_adj_idx]
                term_list = [core_term]

                if dependency[core_adj_idx].relation == Dependency.CMP.value:
                    # 往前找，找名词搭配形容词
                    for n_idx in range(max(0, core_adj_idx - 1), -1, -1):
                        term = segment[n_idx]
                        pos = postag[n_idx]
                        if pos.startswith("n"):
                            pos = Pos.n.value
                        relation = dependency[n_idx].relation
                        if term in negative_feature_words and n_idx < core_adj_idx:
                            term_list.insert(0, term)
                            core_term_idx += 1
                        if pos == Pos.n.value and relation != Dependency.DBL.value and n_idx < core_adj_idx:
                            term_list.insert(0, term)
                            core_term_idx += 1
                            match = True
                            break
                    if not match:
                        # 往前找，找动词搭配形容词
                        term_list = [core_term]
                        for v_idx in range(max(0, core_adj_idx - 1), -1, -1):
                            term = segment[v_idx]
                            if term in negative_feature_words and n_idx < core_adj_idx:
                                term_list.insert(0, term)
                                core_term_idx += 1
                            if postag[v_idx] == Pos.v.value and n_idx < core_adj_idx:
                                term_list.insert(0, term)
                                core_term_idx += 1
                                match = True
                                break
                else:
                    # 找名词搭配形容词
                    next_pos = None
                    next_idx = core_adj_idx + 1
                    if next_idx < length:
                        next_pos = postag[next_idx]
                    search_direction = "before"
                    if next_pos is not None and next_pos == Pos.u.value:
                        if next_idx + 1 < length and postag[next_idx + 1] == Pos.n.value:
                            search_direction = "after"

                    if search_direction == "before":
                        # 往前找
                        for n_idx in range(max(0, core_adj_idx - 1), -1, -1):
                            term = segment[n_idx]
                            pos = postag[n_idx]
                            if pos.startswith("n"):
                                pos = Pos.n.value
                                relation = dependency[n_idx].relation
                            if term in negative_feature_words and n_idx < core_adj_idx:
                                term_list.insert(core_term_idx, term)
                                core_term_idx += 1
                            elif pos == Pos.n.value and relation != Dependency.DBL.value and n_idx < core_adj_idx:
                                term_list.insert(0, term)
                                core_term_idx += 1
                                match = True
                                break
                        # 往后找
                        if not match:
                            term_list = [core_term]
                            for n_idx in range(0, length):
                                term = segment[n_idx]
                                pos = postag[n_idx]
                                if pos.startswith("n"):
                                    pos = Pos.n.value
                                relation = dependency[n_idx].relation
                                if term in negative_feature_words and n_idx < core_adj_idx:
                                    term_list.insert(len(term_list) - 1, term)
                                elif pos == Pos.n.value and relation != Dependency.DBL.value and n_idx > core_adj_idx:
                                    term_list.insert(0, term)
                                    core_term_idx += 1
                                    # term_list.append(term)
                                    match = True
                                    break
                    else:
                        # 往后找
                        for n_idx in range(0, length):
                            term = segment[n_idx]
                            pos = postag[n_idx]
                            if pos.startswith("n"):
                                pos = Pos.n.value
                            relation = dependency[n_idx].relation
                            if term in negative_feature_words and n_idx < core_adj_idx:
                                term_list.insert(len(term_list) - 1, term)
                            elif pos == Pos.n.value and relation != Dependency.DBL.value and n_idx > core_adj_idx:
                                term_list.insert(0, term)
                                core_term_idx += 1
                                match = True
                                break
                # 形容词的以上规则没找到则找否定词
                if not match:
                    term_list = [core_term]
                    for negative_idx in range(core_adj_idx, -1, -1):
                        term = segment[negative_idx]
                        if term in negative_feature_words and negative_idx < core_adj_idx:
                            term_list.insert(len(term_list) - 1, term)
                            match = True

                if not match:
                    term_list = [core_term]
                break

        # if not match:
        #     # 找名词
        #     for idx, pos in enumerate(postag):
        #         term = segment[idx]
        #         pos = postag[n_idx]
        #         if pos.startswith("n"):
        #             pos = Pos.n.value
        #         relation = dependency[idx].relation
        #         # i 类型的词要小于等于2个字
        #         if pos == Pos.n.value and relation in [Dependency.SBV.value, Dependency.FOB.value, Dependency.VOB.value]:
        #             core_n_idx = idx
        #             term_list.append(term)
        #             core_term_idx = 0
        #             # 检测下个词是否是名词
        #             new_core_n_idx = core_n_idx
        #             for next_idx in range(min(length, core_n_idx + 1), length):
        #                 next_pos = postag[next_idx]
        #                 if next_pos.startswith("n"):
        #                     next_pos = Pos.n.value
        #                 if next_pos == Pos.n.value and new_core_n_idx + 1 == next_idx:
        #                     term = segment[next_idx]
        #                     new_core_n_idx = next_idx
        #                     term_list.pop(0)
        #                     term_list.append(term)
        #                 else:
        #                     break
        #             core_n_idx = new_core_n_idx
        #             # 往后找动词
        #             if dependency[new_core_n_idx].relation in [Dependency.SBV.value, Dependency.FOB.value]:
        #                 for n in range()
        #             # 往前找
        #             else:

        # if not match:
        #     # 找动词
        #     for idx, pos in enumerate(postag):
        #         term = segment[idx]
        #         relation = dependency[idx].relation
        #         # i 类型的词要小于等于2个字
        #         if pos == Pos.a.value or (pos == Pos.i.value and len(term) <= 2):
        #             core_adj_idx = idx
        #             term_list.append(term)
        #             core_term_idx = 0
        #             # 检测下个词是否是形容词
        #             new_core_adj_idx = core_adj_idx
        #             for next_idx in range(min(length, core_adj_idx + 1), length):
        #                 if postag[next_idx] == Pos.a.value and new_core_adj_idx + 1 == next_idx:
        #                     term = segment[next_idx]
        #                     new_core_adj_idx = next_idx
        #                     term_list.pop(0)
        #                     term_list.append(term)
        #                 else:
        #                     break

    return term_list


def doc_to_vector(segment, embedding):
    doc_vector = np.zeros(300)
    for term in segment:
        term_vector = np.zeros(300)
        if term in embedding.vocab:
            term_vector = embedding[term]
        doc_vector += term_vector
    doc_vector = doc_vector / len(segment)
    return doc_vector


def cosin_distance(segment1, segment2, embedding):
    """
    计算两个向量的余弦距离
    :param segment1:
    :param segment2:
    :return:
    """
    if "".join(segment1) == "".join(segment2):
        return 1.00001
    vector1 = doc_to_vector(segment1, embedding)
    vector2 = doc_to_vector(segment2, embedding)
    cos_distance = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * (np.linalg.norm(vector2)))
    return cos_distance


def cluster_which_rule(docs):
    doc_segment_list = [[term for term in segmentor.segment(doc)] for doc in docs]
    doc_postag_list = [[pos for pos in postagger.postag(segment)] for segment in doc_segment_list]

    all_term_list = []
    pos_count_dict = {}
    corpora = []
    for doc_idx, doc_segment in enumerate(doc_segment_list):
        doc_pos = doc_postag_list[doc_idx]
        term_list = extract_which_rule(doc_segment, doc_pos, None)
        all_term_list += term_list
        for term, pos in term_list:
            if not pos_count_dict.__contains__(pos):
                pos_count_dict[pos] = 0
            pos_count_dict[pos] += 1

    main_pos = None
    main_pos_count = 0

    for pos, count in pos_count_dict.items():
        if main_pos is None:
            main_pos = pos
            main_pos_count = count
        else:
            if count > main_pos_count:
                main_pos = pos
                main_pos_count = count

    main_term_list = []
    for term, pos in all_term_list:
        if pos == main_pos:
            main_term_list.append(term)

    corpora = [" ".join(main_term_list)]
    tfidf_model = TfidfVectorizer(analyzer='word', token_pattern=r"(?u)\b\w+\b")
    tfidf = tfidf_model.fit_transform(corpora)
    cbow = tfidf_model.get_feature_names()
    return cbow


def cluster_why_rule(docs):
    UNKOWN_FEATURE = "unkown_feature"
    cluster = {}
    print("load embedding")
    embedding = KeyedVectors.load_word2vec_format("/Users/zhuzhibin/Program/python/qd/nlp/nlp-platform/opinion-data/embedding/min.sgns.baidubaike.bigram-char.model", binary=False)

    doc_segment_list = [[term for term in segmentor.segment(doc)] for doc in docs]
    doc_postag_list = [[pos for pos in postagger.postag(segment)] for segment in doc_segment_list]

    # 按特征聚类
    feature_cluster = {}
    for doc_idx, doc_postag in enumerate(doc_postag_list):
        doc_segment = doc_segment_list[doc_idx]
        classify = False
        # 查找特征
        for pos_idx, pos in enumerate(doc_postag):
            term = doc_segment[pos_idx]
            if pos in [Pos.nh.value, Pos.j.value, Pos.nz.value, Pos.ni.value, Pos.z.value] or pos == Pos.n.value or (pos == Pos.i.value and len(term) > 2):
                if len(feature_cluster) == 0:
                    feature_cluster[term] = [(doc_segment, doc_postag)]
                    classify = True
                    break
                else:
                    for feature, feature_segment_list in feature_cluster.items():
                        if cosin_distance([feature], [term], embedding) > 0.8:
                            feature_segment_list.append((doc_segment, doc_postag))
                            classify = True
                            break
                    if not classify:
                        feature_cluster[term] = [(doc_segment, doc_postag)]
                        classify = True
                        break
                    break
        if not classify:
            if not feature_cluster.__contains__(UNKOWN_FEATURE):
                feature_cluster[UNKOWN_FEATURE] = []
            feature_cluster[UNKOWN_FEATURE].append((doc_segment, doc_postag))

    # 按特征观点聚类
    feature_opinion_cluster = {}
    for item in feature_cluster.items():
        feature = item[0]
        feature_segment_postag_list = item[1]
        for doc_segment, doc_postag in feature_segment_postag_list:
            classify = False
            for pos_idx, pos in enumerate(doc_postag):
                term = doc_segment[pos_idx]
                if pos == Pos.a.value or (pos == Pos.i.value and len(term) <= 2):
                    if len(feature_opinion_cluster) == 0:
                        feature_opinion_cluster[str(feature + "|" + term)] = [(doc_segment, doc_postag)]
                        classify = True
                        break
                    else:
                        for feature_opinion, feature_opinion_segment_list in feature_opinion_cluster.items():
                            if cosin_distance(feature_opinion.split("|"), [feature, term], embedding) > 0.8:
                                feature_opinion_segment_list.append((doc_segment, doc_postag))
                                classify = True
                                break
                        if not classify:
                            feature_opinion_cluster[str(feature + "|" + term)] = [(doc_segment, doc_postag)]
                            classify = True
                            break
                        break
            if not classify:
                for feature_opinion, feature_opinion_segment_list in feature_opinion_cluster.items():
                    if cosin_distance(feature_opinion.split("|"), [feature], embedding) > 0.8:
                        feature_opinion_segment_list.append((doc_segment, doc_postag))
                        classify = True
                        break
                if not classify:
                    feature_opinion_cluster[feature] = [(doc_segment, doc_postag)]

    for n, item in enumerate(feature_opinion_cluster.items()):
        doc_segment_list = []
        doc_segment_postag_list = item[1]
        for segment, postag in doc_segment_postag_list:
            doc_segment_list.append("".join(segment))
        cluster[n] = doc_segment_list

    return cluster


# load_corpus()
# # process_corpus()
# model = train_model()
# print(extract_opinion("第一次吃的时候感觉不会特别甜", model))

# show_model(model)
# f1 = compute_f1_score(test, model)
# print(f1)

print(doc_split("本奶粉可以给孩子一个好的未来，让孩子强壮"))
print(doc_split("因为西北部地区缺水相对严重，有的时候看新闻会看到，就顺便关注了一下，毕竟都是自己的同胞，还有水是生命之源嘛大家都会关注点的"))

# print(extract_opinion2("这个个大品牌身边朋友有很多都是选择的这个"))
# embedding = KeyedVectors.load_word2vec_format("/Users/zhuzhibin/Program/python/qd/nlp/nlp-platform/opinion-data/embedding/sgns.baidubaike.bigram-char.model", binary=False)
# print(cosin_distance(["好"], ["不好"], embedding))
# print(cosin_distance(["冷萃"], ["方便"], embedding))
# 蛋糕多是现做的水果多是新鲜的