import unittest

from sentiment.answer_tfidf import CommentParser


class OpinionTest(unittest.TestCase):
    def test_smart_split_doc(self):
        import extract.analysis as aly
        with open("../data/comments/which/comment", "r", encoding="utf-8") as corpus:
            for line in corpus:
                try:
                    sub_doc_list = aly.smart_doc_split(line.strip())
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

    def test_extract_why_opinion(self):
        import extract.analysis as aly
        parser = CommentParser()
        aly.load_corpus()
        model = aly.train_model()
        show_detail = False
        with open("../data/comments/why/comment", "r", encoding="utf-8") as corpus:
            for line in corpus:
                line = line.strip()
                # try:
                print(line)
                opinions = aly.extract_why_opinion(line)
                print("opinions:", opinions)
                sub_doc_list = aly.smart_doc_split(line)
                for sub_doc in sub_doc_list:
                    if show_detail:
                        parser.sentence_segment_ltp(sub_doc)
                print("====================================================")
                print("\n")
                # except Exception:
                #     print("exception: ", line)

    def test_extract_which_opinion(self):
        import extract.analysis as aly
        parser = CommentParser()
        aly.load_corpus()
        show_detail = False
        with open("../data/comments/which/comment", "r", encoding="utf-8") as corpus:
            for line in corpus:
                line = line.strip()
                # try:
                print(line)
                opinions = aly.extract_which_opinion(line)
                print("opinions:", opinions)
                sub_doc_list = aly.smart_doc_split(line)
                for sub_doc in sub_doc_list:
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
        print(aly.cluster_why_rule(opinion_list))

    def test_cluster_which_opinion(self):
        import extract.analysis as aly
        with open("../data/comments/which/comment", "r", encoding="utf-8") as corpus:
            cbow = aly.cluster_which_rule(corpus)
            for word in cbow:
                print(word, ":", cbow.count(word))

    def test_ltp_sentence_splitter(self):
        from pyltp import SentenceSplitter
        splitter = SentenceSplitter()
        docs = splitter.split("爱他美可以给孩子一个健康美好的未来，能给孩子全面均衡的营养。")
        for doc in docs:
            print(doc)