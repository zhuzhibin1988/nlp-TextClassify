import jieba
import jieba.posseg as jb_posseg  # 词性标注
import jieba.analyse as jb_analyse  # 关键词提取
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import numpy as np

"""
1.答案文档分词
2.构造答案词典
3.训练词典单词的tfidf(w)=tf(d,w)*idf(w)
4.计算句子的tfidf vector
5.关键词提取
"""


def stopwordlist(*self_define_stopwords):
    """
    get stopwords list
    :param self_define_stopwords: add self define stop word to stopwords list
    :return: stopwordlist
    """
    stopwords = [word.strip() for word in open("../data/stopwords", "r").readlines()]
    for stopword in self_define_stopwords:
        stopwords.append(stopword)
    return stopwords


def sentence_segment(comment, stopword_list={}):
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


def sentence_segment_by_bland(comment, stopword_list={}):
    """
    使用空格间隔分词
    如：
    我们 喜欢 吃 冰激凌
    :param comment: 一条语料
    :param stopword_list: 停用词列表
    :return:
    """
    segment = sentence_segment(comment, stopword_list)
    return " ".join(segment)


def construct_pow(use_stopwords):
    """
    构建词袋
    :param use_stopwords:
    :return:
    """
    stopword_list = {}
    if use_stopwords:
        stopword_list = stopwordlist("之前", "是因为", "已经")
    dictionary = set()
    with open("../data/comment", "r") as comments:
        for comment in comments:
            segment = sentence_segment(comment, stopword_list)
            for word in segment:
                dictionary.add(word)
    print(dictionary)


def compute_tfidf():
    """
    计算词袋里的词的tfidf
    :return:
    """
    corpus = []
    tfidfdict = {}
    with open("../data/comment", "r") as comments:
        for comment in comments:
            corpus.append(sentence_segment_by_bland(comment))
    vectorizer = CountVectorizer(stop_words=stopwordlist("之前", "是因为", "已经"))
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


def self_define():
    jb_analyse.set_stop_words("../data/stopwords")
    jb_analyse.set_idf_path("../data/sk_tfidf.txt")


# construct_pow(True)
compute_tfidf()