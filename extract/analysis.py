import random
import os
from pyltp import Segmentor, Postagger, Parser, SementicRoleLabeller
from sentiment.pos import Pos
from sentiment.dependency import Dependency

LTP_MODEL_DIR = "E:/02program/python/nlp/data/ltp_data_v3.4.0"


def process_corpus():
    segmentor = Segmentor()
    postagger = Postagger()
    parser = Parser()  # 初始化实例
    labeller = SementicRoleLabeller()  # 初始化实例

    segmentor.load(os.path.join(LTP_MODEL_DIR, "cws.model"))
    postagger.load(os.path.join(LTP_MODEL_DIR, "pos.model"))
    parser.load(os.path.join(LTP_MODEL_DIR, "parser.model"))  # 加载模型
    labeller.load(os.path.join(LTP_MODEL_DIR, "pisrl_win.model"))  # 加载模型

    seq_dict = {}
    with open("../data/tag-data/why-tag.txt", "r", encoding="utf-8") as tag_data:
        lines = [line for line in tag_data.readlines()]
        size = len(lines)
        train_size = int(size * 0.8)
        test_size = size - train_size
        print("train size: ", train_size, "test_size: ", test_size)

        train_lines = [lines[idx].strip() for idx in random.sample(range(0, size), train_size)]
        for train_line in train_lines:
            doc = train_line.split(":")[0]
            segment = segmentor.segment(doc)
            postag = postagger.postag(segment)
            dependency = parser.parse(segment, postag)
            seq_list = []
            for idx, node in enumerate(dependency):
                word = segment[idx]
                pos = postag[idx]
                if pos.startswith("n"):
                    pos = Pos.n.value
                relation = node.relation
                if relation not in [Dependency.ADV.value, Dependency.ATT.value]:
                    seq_list.append(pos)
                    seq_list.append(relation)
                elif relation == Dependency.ADV.value and word == "不":
                    seq_list.append(pos)
                    seq_list.append("不")
                    seq_list.append(relation)
            seq = "-".join(seq_list)
            if seq_dict.__contains__(seq):
                seq_dict[seq] += 1
            else:
                seq_dict[seq] = 1

    for item in seq_dict.items():
        print(item)


process_corpus()
