import random
import re
import os


def prepare_dataset():
    comment_dataset = []
    with open("../data/corpus/type_why", "r", encoding="utf-8") as all:
        size = int(all.readline())
        print(size)
        random_comment_ids = random.sample(range(1, size + 1), 1000)
        comments = all.readlines()
        for id in random_comment_ids:
            comment = comments[id - 1]
            comment = re.sub(re.compile(r"(\s+)", re.S), "，", comment.strip())
            # 句子按分隔[。|！|，|、|？|.|!|,|?]符分出多个子句
            subcomments = re.split(r'[。|！|，|、|？|\.|!|,|\?]', comment)
            for subcomment in subcomments:
                if len(subcomment) > 0:
                    slip_list = []
                    slip(subcomment, slip_list)
                    comment_dataset += [s + os.linesep for s in slip_list]

    with open("../data/dataset/ds_why", "w", encoding="utf-8") as ds:
        ds.writelines(comment_dataset)


def slip(comment, tmp_list):
    """
    并列句拆解

    既。。。又。。。
    :param comment:
    :param tmp_list:
    :return:
    """
    keywords = ["且", "并且", "而且", "还", "还有", "和"]
    can_slip = False
    for keyword in keywords:
        position = comment.rfind(keyword)
        if position > -1:
            can_slip = True
            left = comment[0:position - 1].strip()
            if len(left) > 0:
                tmp_list.append(left)
            right = comment[position + len(keyword):len(comment)].strip()
            slip(right, tmp_list)
            break
    if not can_slip:
        if len(comment.strip()) > 0:
            tmp_list.append(comment.strip())


# comment_list = []
# slip("并最好有洗拖功能的", comment_list)
# print(comment_list)

prepare_dataset()