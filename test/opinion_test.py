import unittest

from sentiment.answer_tfidf import CommentParser


class OpinionTest(unittest.TestCase):
    def test_smart_split_doc(self):
        import extract.analysis as aly
        with open("../data/comment3", "r", encoding="utf-8") as corpus:
            for line in corpus:
                try:
                    sub_doc_list = aly.smart_split_doc(line.strip())
                    for sub_doc in sub_doc_list:
                        print(sub_doc)
                        segment = aly.segmentor.segment(sub_doc)
                        postag = aly.postagger.postag(segment)
                        segment = [term for term in segment]
                        postag = [pos for pos in postag]

                        dependency = aly.parser.parse(segment, postag)
                        term_pos_list = []
                        for n, term in enumerate(segment):
                            term_pos_list.append(term + "(" + postag[n] + "-" + dependency[n].relation + ")")
                        print("/".join(term_pos_list))
                except Exception:
                    print("exception: ", line.strip())

    def test_extract_opinion(self):
        import extract.analysis as aly
        parser = CommentParser()
        aly.load_corpus()
        model = aly.train_model()
        show_detail = False
        with open("../data/comment", "r", encoding="utf-8") as corpus:
            for line in corpus:
                line = line.strip()
                # try:
                print(line)
                sub_doc_list = aly.smart_split_doc(line)
                for sub_doc in sub_doc_list:
                    opinion = aly.extract_opinion3(sub_doc)
                    print(sub_doc, ":", opinion)
                    if show_detail:
                        parser.sentence_segment_ltp(sub_doc)
                        print("====================================================")
                print("\n")
                # except Exception:
                #     print("exception: ", line)

    def test_cluster_opinion(self):
        import extract.analysis as aly
        opinion_list = []
        with open("../data/comment1", "r", encoding="utf-8") as corpus:
            for line in corpus:
                line = line.strip()
                opinion_list += aly.extract_opinion3(line)
        print(aly.cluster_rule(opinion_list))